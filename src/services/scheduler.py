"""Local scheduler for running the AI CTO Daily pipeline as a service."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

from ..models import ScheduleConfig


ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_RUNS_DIR = ROOT / "data" / "scheduler-runs"


JobStarter = Callable[[int | None], dict[str, Any]]
ScheduleLoader = Callable[[], ScheduleConfig]


@dataclass
class SchedulerState:
    running: bool = False
    next_run_at: datetime | None = None
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_job_id: str | None = None
    last_job: dict[str, Any] | None = None
    last_error: str | None = None
    current_job_running: bool = False
    run_on_start_done: bool = False
    schedule_signature: str | None = None


def parse_daily_time(value: str) -> dt_time:
    hour_text, minute_text = value.split(":", 1)
    return dt_time(hour=int(hour_text), minute=int(minute_text))


def schedule_signature(config: ScheduleConfig) -> str:
    return config.model_dump_json()


def compute_next_run(
    config: ScheduleConfig,
    now: datetime | None = None,
    *,
    last_started_at: datetime | None = None,
) -> datetime | None:
    """Return the next scheduled run time in UTC."""
    if not config.enabled:
        return None

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    if config.interval_minutes is not None:
        base = last_started_at.astimezone(timezone.utc) if last_started_at else now
        candidate = base + timedelta(minutes=config.interval_minutes)
        if candidate <= now:
            candidate = now + timedelta(seconds=1)
        return candidate

    tz = ZoneInfo(config.timezone)
    local_now = now.astimezone(tz)
    times = sorted(parse_daily_time(value) for value in config.daily_times)
    for scheduled_time in times:
        candidate_local = datetime.combine(local_now.date(), scheduled_time, tzinfo=tz)
        if candidate_local > local_now:
            return candidate_local.astimezone(timezone.utc)

    tomorrow = local_now.date() + timedelta(days=1)
    return datetime.combine(tomorrow, times[0], tzinfo=tz).astimezone(timezone.utc)


class PipelineScheduler:
    """Background scheduler that triggers pipeline runs from config."""

    def __init__(
        self,
        load_schedule: ScheduleLoader,
        start_job: JobStarter,
        *,
        poll_seconds: float = 30.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.load_schedule = load_schedule
        self.start_job = start_job
        self.poll_seconds = poll_seconds
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.state = SchedulerState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self.state.running:
                return
            self._stop.clear()
            self.state.running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self.state.running = False
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def trigger_now(self, hours: int | None = None) -> dict[str, Any]:
        config = self.load_schedule()
        return self._trigger(hours if hours is not None else config.hours)

    def status(self) -> dict[str, Any]:
        config = self._safe_load_schedule()
        with self._lock:
            next_run_at = self.state.next_run_at
            return {
                "service_running": self.state.running,
                "enabled": config.enabled if config else False,
                "config": config.model_dump(mode="json") if config else None,
                "next_run_at": next_run_at.isoformat() if next_run_at else None,
                "last_started_at": (
                    self.state.last_started_at.isoformat()
                    if self.state.last_started_at
                    else None
                ),
                "last_finished_at": (
                    self.state.last_finished_at.isoformat()
                    if self.state.last_finished_at
                    else None
                ),
                "last_job_id": self.state.last_job_id,
                "last_job": self.state.last_job,
                "last_error": self.state.last_error,
                "current_job_running": self.state.current_job_running,
            }

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                config = self.load_schedule()
                signature = schedule_signature(config)
                now = self.now().astimezone(timezone.utc)

                with self._lock:
                    if signature != self.state.schedule_signature:
                        self.state.schedule_signature = signature
                        self.state.next_run_at = compute_next_run(
                            config,
                            now,
                            last_started_at=self.state.last_started_at,
                        )

                    should_run_on_start = (
                        config.enabled
                        and config.run_on_start
                        and not self.state.run_on_start_done
                    )

                if should_run_on_start:
                    self._trigger(config.hours)
                    with self._lock:
                        self.state.run_on_start_done = True
                        self.state.next_run_at = compute_next_run(
                            config,
                            self.now(),
                            last_started_at=self.state.last_started_at,
                        )

                with self._lock:
                    next_run_at = self.state.next_run_at

                if config.enabled and next_run_at and now >= next_run_at:
                    self._trigger(config.hours)
                    with self._lock:
                        self.state.next_run_at = compute_next_run(
                            config,
                            self.now() + timedelta(seconds=1),
                            last_started_at=self.state.last_started_at,
                        )

                sleep_for = self._sleep_seconds()
            except Exception as exc:
                with self._lock:
                    self.state.last_error = str(exc)
                sleep_for = self.poll_seconds

            self._stop.wait(max(0.5, sleep_for))

    def _trigger(self, hours: int | None) -> dict[str, Any]:
        with self._lock:
            config = self.load_schedule()
            if config.prevent_overlap and self.state.current_job_running:
                return {
                    "started": False,
                    "reason": "A scheduled job is already running.",
                    "last_job_id": self.state.last_job_id,
                }
            self.state.current_job_running = True
            self.state.last_started_at = self.now().astimezone(timezone.utc)
            self.state.last_error = None

        try:
            job = self.start_job(hours)
            with self._lock:
                self.state.last_job_id = str(job.get("id") or job.get("job_id") or "")
                self.state.last_job = job
            return {"started": True, "job": job}
        except Exception as exc:
            with self._lock:
                self.state.last_error = str(exc)
            return {"started": False, "error": str(exc)}
        finally:
            with self._lock:
                self.state.current_job_running = False
                self.state.last_finished_at = self.now().astimezone(timezone.utc)

    def _sleep_seconds(self) -> float:
        with self._lock:
            next_run_at = self.state.next_run_at
        if not next_run_at:
            return self.poll_seconds
        delta = (next_run_at - self.now().astimezone(timezone.utc)).total_seconds()
        return min(self.poll_seconds, max(0.5, delta))

    def _safe_load_schedule(self) -> ScheduleConfig | None:
        try:
            return self.load_schedule()
        except Exception as exc:
            with self._lock:
                self.state.last_error = str(exc)
            return None


def start_blocking_cli_job(hours: int | None) -> dict[str, Any]:
    """Run the normal CLI pipeline and block until it exits."""
    SCHEDULER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"scheduled-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    job_dir = SCHEDULER_RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"

    command = [sys.executable, "-m", "src.main"]
    if hours:
        command.extend(["--hours", str(hours)])

    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return {
        "id": job_id,
        "status": "success" if process.returncode == 0 else "error",
        "returncode": process.returncode,
        "command": command,
        "log_path": str(log_path),
    }


def run_forever(load_schedule: ScheduleLoader, *, poll_seconds: float = 30.0) -> None:
    scheduler = PipelineScheduler(
        load_schedule,
        start_blocking_cli_job,
        poll_seconds=poll_seconds,
    )
    scheduler.start()
    try:
        while True:
            status = scheduler.status()
            next_run = status.get("next_run_at") or "not scheduled"
            print(f"AI CTO Daily scheduler running. Next run: {next_run}", flush=True)
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping scheduler...", flush=True)
    finally:
        scheduler.stop()
