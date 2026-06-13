"""Local AI CTO Daily web UI.

This intentionally uses only Python's standard library so the UI can be
started in the same environment as the CLI without a frontend build step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import ValidationError
import markdown

from ..models import Config
from ..publishers.wechat import WeChatPublisher
from ..publishers.wechat_renderer import (
    append_full_version_note,
    extract_digest,
    extract_title,
    markdown_to_wechat_html,
)
from ..storage.manager import ConfigError, StorageManager, _expand_env_vars
from ..services.scheduler import PipelineScheduler, start_blocking_cli_job
from ..ai import prompts as default_prompts


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_PATH = DATA_DIR / "config.json"
SUMMARIES_DIR = DATA_DIR / "summaries"
ARTIFACT_RUNS_DIR = DATA_DIR / "runs"
WEB_RUNS_DIR = DATA_DIR / "web-runs"
WECHAT_RECORDS_DIR = DATA_DIR / "publishing" / "wechat"


@dataclass
class WebJob:
    id: str
    command: list[str]
    created_at: str
    log_path: Path
    topic_slug: str | None = None
    status: str = "running"
    returncode: int | None = None
    finished_at: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)


JOBS: dict[str, WebJob] = {}
JOBS_LOCK = threading.Lock()
SCHEDULER_LOCK = threading.Lock()
SCHEDULERS: dict[str, PipelineScheduler] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _load_config_raw() -> str:
    if not CONFIG_PATH.exists():
        return "{}\n"
    return CONFIG_PATH.read_text(encoding="utf-8-sig")


PROMPT_DEFAULTS: dict[str, str] = {
    "analysis_system": default_prompts.build_content_analysis_system(),
    "analysis_user": default_prompts.CONTENT_ANALYSIS_USER,
    "concept_system": default_prompts.CONCEPT_EXTRACTION_SYSTEM,
    "concept_user": default_prompts.CONCEPT_EXTRACTION_USER,
    "enrichment_system": default_prompts.CONTENT_ENRICHMENT_SYSTEM,
    "enrichment_user": default_prompts.CONTENT_ENRICHMENT_USER,
    "takeaway_system": default_prompts.CTO_TAKEAWAY_SYSTEM,
    "takeaway_user": default_prompts.CTO_TAKEAWAY_USER,
    "topic_dedup_system": default_prompts.TOPIC_DEDUP_SYSTEM,
    "topic_dedup_user": default_prompts.TOPIC_DEDUP_USER,
    "translation_system": default_prompts.TRANSLATION_FALLBACK_SYSTEM,
    "translation_user": default_prompts.TRANSLATION_FALLBACK_USER,
}


def _fill_default_prompts_for_editor(data: dict[str, Any]) -> dict[str, Any]:
    """Return a config copy with empty topic prompts expanded for editing."""
    data = json.loads(json.dumps(data))
    for topic in data.get("topics", []) or []:
        if not isinstance(topic, dict):
            continue
        prompts = topic.setdefault("prompts", {})
        if not isinstance(prompts, dict):
            prompts = {}
            topic["prompts"] = prompts
        topic_defaults = dict(PROMPT_DEFAULTS)
        try:
            topic_model = Config.model_validate({**data, "active_topic": topic.get("slug")}).get_active_topic()
            topic_defaults["analysis_system"] = default_prompts.build_content_analysis_system(topic_model)
        except Exception:
            pass
        for key, value in topic_defaults.items():
            if not prompts.get(key):
                prompts[key] = value
    return data


def _config_text_for_editor() -> str:
    data = _parse_config_text(_load_config_raw())
    return json.dumps(_fill_default_prompts_for_editor(data), indent=2, ensure_ascii=False) + "\n"


def _parse_config_text(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError("Config must be a JSON object.")
    return parsed


def _validate_config_text(text: str) -> Config:
    data = _parse_config_text(text)
    try:
        return Config.model_validate(_expand_env_vars(data))
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def _save_config_text(text: str) -> Path:
    data = _parse_config_text(text)
    _validate_config_text(text)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        backup_path = CONFIG_PATH.with_suffix(".json.bak")
        backup_path.write_text(CONFIG_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return CONFIG_PATH


def _job_payload(job: WebJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        "returncode": job.returncode,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "command": job.command,
        "log_path": str(job.log_path),
        "topic": job.topic_slug,
    }


def _safe_child(root: Path, raw: str) -> Path:
    if not raw:
        raise ValueError("Missing path segment")
    path = (root / unquote(raw)).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError("Invalid path")
    return path


def _list_summaries(topic_slug: str | None = None) -> list[dict[str, Any]]:
    if not SUMMARIES_DIR.exists():
        return []
    items = []
    for path in sorted(SUMMARIES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        if topic_slug and not path.name.startswith(f"{topic_slug}-"):
            continue
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return items


def _list_artifact_runs(topic_slug: str | None = None) -> list[dict[str, Any]]:
    if not ARTIFACT_RUNS_DIR.exists():
        return []
    items = []
    for path in sorted(ARTIFACT_RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir() or not path.name.startswith("run-"):
            continue
        meta_path = path / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        run_topic = meta.get("topic_slug")
        if topic_slug and run_topic and run_topic != topic_slug:
            continue
        stages_dir = path / "stages"
        summaries_dir = path / "summaries"
        stages = sorted(p.stem for p in stages_dir.glob("*.json")) if stages_dir.exists() else []
        summaries = sorted(p.name for p in summaries_dir.glob("*.md")) if summaries_dir.exists() else []
        ai_calls = path / "ai_calls.jsonl"
        stat = path.stat()
        items.append(
            {
                "id": path.name,
                "topic_slug": run_topic,
                "topic_name": meta.get("topic_name"),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "stages": stages,
                "summaries": summaries,
                "has_ai_calls": ai_calls.exists(),
            }
        )
    return items


def _render_markdown(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=["extra", "sane_lists", "toc"],
        output_format="html5",
    )


def _config_with_topic(config: Config, topic_slug: str | None) -> Config:
    if not topic_slug:
        return config
    topic_slug = topic_slug.strip()
    if not topic_slug:
        return config
    data = config.model_dump(mode="json")
    data["active_topic"] = topic_slug
    return Config.model_validate(data)


def _topic_slug_from_mapping(data: dict[str, Any] | None) -> str | None:
    if not data:
        return None
    value = data.get("topic")
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value).strip() if value else None


def _load_config_model(topic_slug: str | None = None) -> Config:
    load_dotenv(ROOT / ".env", override=False)
    storage = StorageManager(data_dir=str(DATA_DIR))
    return _config_with_topic(storage.load_config(), topic_slug)


def _load_schedule_config(topic_slug: str | None = None):
    return _load_config_model(topic_slug).schedule


def _get_scheduler(topic_slug: str | None = None) -> PipelineScheduler:
    config = _load_config_model(topic_slug)
    slug = config.get_active_topic().slug
    with SCHEDULER_LOCK:
        if slug not in SCHEDULERS:
            SCHEDULERS[slug] = PipelineScheduler(
                lambda slug=slug: _load_schedule_config(slug),
                lambda hours, slug=slug: start_blocking_cli_job(hours, slug),
                poll_seconds=10.0,
            )
        return SCHEDULERS[slug]


def _scheduler_status(topic_slug: str | None = None) -> dict[str, Any]:
    config = _load_config_model(topic_slug)
    topic = config.get_active_topic()
    status = _get_scheduler(topic.slug).status()
    status["topic"] = {
        "slug": topic.slug,
        "name": topic.name,
        "title_en": topic.title_en,
        "title_zh": topic.title_zh,
    }
    return status


def _read_summary_by_name(name: str) -> str:
    file_path = _safe_child(SUMMARIES_DIR, name)
    if not file_path.exists() or file_path.suffix.lower() != ".md":
        raise FileNotFoundError(f"Summary not found: {name}")
    return file_path.read_text(encoding="utf-8")


def _wechat_status(topic_slug: str | None = None) -> dict[str, Any]:
    config = _load_config_model(topic_slug)
    wc = config.publishing.wechat
    appid_present = bool(os.getenv(wc.appid_env))
    secret_present = bool(os.getenv(wc.secret_env))
    cover_path = Path(wc.cover_image).expanduser()
    if not cover_path.is_absolute():
        cover_path = ROOT / cover_path
    return {
        "enabled": wc.enabled,
        "appid_env": wc.appid_env,
        "secret_env": wc.secret_env,
        "appid_present": appid_present,
        "secret_present": secret_present,
        "author": wc.author,
        "cover_image": wc.cover_image,
        "cover_exists": cover_path.exists(),
        "cover_size": cover_path.stat().st_size if cover_path.exists() else None,
        "default_digest": wc.default_digest,
        "publish_mode": wc.publish_mode,
    }


def _validate_runtime(topic_slug: str | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=False)
    storage = StorageManager(data_dir=str(DATA_DIR))
    config = _config_with_topic(storage.load_config(), topic_slug)
    topic = config.get_active_topic()
    scoped = config.scoped_to_active_topic()
    missing_env = []
    warnings = []
    if config.ai.api_key_env and not os.getenv(config.ai.api_key_env):
        missing_env.append(config.ai.api_key_env)
    if scoped.sources.github and not os.getenv("GITHUB_TOKEN"):
        warnings.append("GITHUB_TOKEN is not set; GitHub may return 401 or rate-limit.")
    if config.webhook and config.webhook.enabled and config.webhook.url_env and not os.getenv(config.webhook.url_env):
        missing_env.append(config.webhook.url_env)
    return {
        "ok": not missing_env,
        "topic": {
            "slug": topic.slug,
            "name": topic.name,
            "title_en": topic.title_en,
            "title_zh": topic.title_zh,
        },
        "ai": {
            "provider": config.ai.provider.value,
            "model": config.ai.model,
            "languages": config.ai.languages,
            "api_key_env": config.ai.api_key_env,
        },
        "filtering": scoped.filtering.model_dump(mode="json"),
        "summary": scoped.summary.model_dump(mode="json"),
        "artifacts": config.artifacts.model_dump(mode="json"),
        "missing_env": missing_env,
        "warnings": warnings,
    }


def _start_cli_job(hours: int | None, topic_slug: str | None = None) -> dict[str, Any]:
    WEB_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    job_dir = WEB_RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"

    command = [sys.executable, "-m", "src.main"]
    if hours:
        command.extend(["--hours", str(hours)])
    config = _load_config_model(topic_slug)
    topic = config.get_active_topic()
    command.extend(["--topic", topic.slug])

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TERM", "xterm")

    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = WebJob(
        id=job_id,
        command=command,
        created_at=_utc_now(),
        log_path=log_path,
        topic_slug=topic.slug,
        process=process,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job

    def _watch() -> None:
        try:
            rc = process.wait()
        finally:
            log_handle.close()
        with JOBS_LOCK:
            job.returncode = rc
            job.finished_at = _utc_now()
            job.status = "success" if rc == 0 else "error"

    threading.Thread(target=_watch, daemon=True).start()
    return _job_payload(job)


class AICTODailyWebHandler(BaseHTTPRequestHandler):
    server_version = "AICTODailyWeb/0.1"

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            topic_slug = _topic_slug_from_mapping(query)
            if path == "/":
                self._send_html(INDEX_HTML)
            elif path == "/api/status":
                self._send_json(self._status_payload(topic_slug))
            elif path == "/api/config":
                self._send_json({"path": str(CONFIG_PATH), "text": _config_text_for_editor()})
            elif path == "/api/summaries":
                self._send_json({"items": _list_summaries(topic_slug)})
            elif path.startswith("/api/summaries/"):
                name = path.removeprefix("/api/summaries/")
                file_path = _safe_child(SUMMARIES_DIR, name)
                text = file_path.read_text(encoding="utf-8")
                self._send_json({"name": file_path.name, "text": text, "html": _render_markdown(text)})
            elif path == "/api/artifact-runs":
                self._send_json({"items": _list_artifact_runs(topic_slug)})
            elif path == "/api/publish/wechat/status":
                self._send_json(_wechat_status(topic_slug))
            elif path == "/api/schedule/status":
                self._send_json(_scheduler_status(topic_slug))
            elif path.startswith("/api/artifact-runs/"):
                self._artifact_route(path)
            elif path == "/api/jobs":
                with JOBS_LOCK:
                    jobs = [_job_payload(job) for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]
                self._send_json({"items": jobs})
            elif path.startswith("/api/jobs/") and path.endswith("/log"):
                job_id = path.split("/")[3]
                self._send_job_log(job_id, query)
            elif path.startswith("/api/jobs/"):
                job_id = path.split("/")[3]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if not job:
                    self._send_error(HTTPStatus.NOT_FOUND, "Job not found")
                else:
                    self._send_json(_job_payload(job))
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            payload = self._read_json()
            topic_slug = _topic_slug_from_mapping(payload)
            if parsed.path == "/api/config/validate":
                config = _validate_config_text(str(payload.get("text", "")))
                self._send_json({"ok": True, "config": config.model_dump(mode="json")})
            elif parsed.path == "/api/config":
                text = str(payload.get("text", ""))
                saved = _save_config_text(text)
                self._send_json({"ok": True, "path": str(saved), "backup": str(saved.with_suffix(".json.bak"))})
            elif parsed.path == "/api/run":
                hours = payload.get("hours")
                hours_int = int(hours) if hours not in (None, "", 0) else None
                if hours_int is not None and hours_int <= 0:
                    raise ValueError("hours must be positive")
                self._send_json(_start_cli_job(hours_int, topic_slug), status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/schedule/start":
                _get_scheduler(topic_slug).start()
                self._send_json(_scheduler_status(topic_slug))
            elif parsed.path == "/api/schedule/stop":
                _get_scheduler(topic_slug).stop()
                self._send_json(_scheduler_status(topic_slug))
            elif parsed.path == "/api/schedule/run-now":
                hours = payload.get("hours")
                hours_int = int(hours) if hours not in (None, "", 0) else None
                if hours_int is not None and hours_int <= 0:
                    raise ValueError("hours must be positive")
                scheduler = _get_scheduler(topic_slug)
                thread = threading.Thread(
                    target=lambda: scheduler.trigger_now(hours_int),
                    daemon=True,
                )
                thread.start()
                self._send_json({"ok": True, "message": "Scheduled run started."}, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/publish/wechat/preview":
                self._wechat_preview(payload)
            elif parsed.path in {"/api/publish/wechat/draft", "/api/publish/wechat/submit"}:
                result = asyncio.run(self._wechat_publish(payload))
                self._send_json(result)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except ConfigError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _status_payload(self, topic_slug: str | None = None) -> dict[str, Any]:
        try:
            validation = _validate_runtime(topic_slug)
        except Exception as exc:
            validation = {"ok": False, "error": str(exc)}
        config = _load_config_model(topic_slug)
        topics = [
            {
                "slug": topic.slug,
                "name": topic.name,
                "title_en": topic.title_en,
                "title_zh": topic.title_zh,
                "enabled": topic.enabled,
            }
            for topic in config.topics
        ]
        active_topic = config.get_active_topic().slug
        with JOBS_LOCK:
            jobs = [_job_payload(job) for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]
        return {
            "root": str(ROOT),
            "config_path": str(CONFIG_PATH),
            "topics": topics,
            "active_topic": active_topic,
            "validation": validation,
            "schedule": _scheduler_status(active_topic),
            "jobs": jobs[:10],
            "summaries": _list_summaries(active_topic)[:8],
            "artifact_runs": _list_artifact_runs(active_topic)[:8],
        }

    def _wechat_preview(self, payload: dict[str, Any]) -> None:
        config = _load_config_model(_topic_slug_from_mapping(payload))
        markdown_text = str(payload.get("text") or "")
        summary_name = str(payload.get("summary_name") or "")
        if not markdown_text and summary_name:
            markdown_text = _read_summary_by_name(summary_name)
        if not markdown_text:
            raise ValueError("summary_name or text is required")

        wc = config.publishing.wechat
        title = str(payload.get("title") or extract_title(markdown_text)).strip()
        digest = str(payload.get("digest") or extract_digest(markdown_text, wc.default_digest)).strip()
        source_url = str(payload.get("content_source_url") or "").strip() or None
        html = markdown_to_wechat_html(append_full_version_note(markdown_text, source_url))
        self._send_json(
            {
                "title": title[:64],
                "digest": digest[:120],
                "author": str(payload.get("author") or wc.author),
                "html": html,
                "text": markdown_text,
            }
        )

    async def _wechat_publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = _load_config_model(_topic_slug_from_mapping(payload))
        wc = config.publishing.wechat
        if not wc.enabled:
            raise ValueError("WeChat publishing is disabled in config.")

        markdown_text = str(payload.get("text") or "")
        summary_name = str(payload.get("summary_name") or "")
        if not markdown_text and summary_name:
            markdown_text = _read_summary_by_name(summary_name)
        if not markdown_text:
            raise ValueError("summary_name or text is required")

        mode = str(payload.get("mode") or wc.publish_mode).strip() or "draft"
        if mode not in {"draft", "publish"}:
            raise ValueError("mode must be draft or publish")

        async with WeChatPublisher(wc, root_dir=ROOT) as publisher:
            result = await publisher.publish_markdown(
                markdown_text,
                title=str(payload.get("title") or "").strip() or None,
                digest=str(payload.get("digest") or "").strip() or None,
                author=str(payload.get("author") or "").strip() or None,
                cover_image=str(payload.get("cover_image") or "").strip() or None,
                content_source_url=str(payload.get("content_source_url") or "").strip() or None,
                record_dir=WECHAT_RECORDS_DIR,
                mode=mode,
            )
        return {
            "ok": True,
            "mode": result.mode,
            "media_id": result.media_id,
            "publish_id": result.publish_id,
            "title": result.title,
            "digest": result.digest,
            "thumb_media_id": result.thumb_media_id,
            "record_path": result.record_path,
        }

    def _artifact_route(self, path: str) -> None:
        rest = path.removeprefix("/api/artifact-runs/")
        parts = rest.split("/")
        if not parts:
            self._send_error(HTTPStatus.NOT_FOUND, "Missing run id")
            return
        run_id = parts[0]
        run_dir = _safe_child(ARTIFACT_RUNS_DIR, run_id)
        if not run_dir.exists() or not run_dir.is_dir():
            self._send_error(HTTPStatus.NOT_FOUND, "Run not found")
            return
        if len(parts) == 1:
            self._send_json({"run": next((r for r in _list_artifact_runs() if r["id"] == run_id), {"id": run_id})})
            return
        kind = parts[1]
        if kind == "stage" and len(parts) >= 3:
            stage_path = _safe_child(run_dir / "stages", f"{parts[2]}.json")
            self._send_json({"run_id": run_id, "stage": parts[2], "items": json.loads(stage_path.read_text(encoding="utf-8"))})
        elif kind == "summary" and len(parts) >= 3:
            summary_path = _safe_child(run_dir / "summaries", parts[2])
            text = summary_path.read_text(encoding="utf-8")
            self._send_json({"run_id": run_id, "name": summary_path.name, "text": text, "html": _render_markdown(text)})
        elif kind == "ai-calls":
            ai_path = run_dir / "ai_calls.jsonl"
            lines = []
            if ai_path.exists():
                limit = 200
                for line in ai_path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        lines.append({"raw": line})
            self._send_json({"run_id": run_id, "items": lines})
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Artifact not found")

    def _send_job_log(self, job_id: str, query: dict[str, list[str]]) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            self._send_error(HTTPStatus.NOT_FOUND, "Job not found")
            return
        tail = int(query.get("tail", ["40000"])[0])
        text = ""
        if job.log_path.exists():
            raw = job.log_path.read_text(encoding="utf-8", errors="replace")
            text = raw[-tail:] if tail > 0 else raw
        self._send_json({"job": _job_payload(job), "log": text})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI CTO Daily Console</title>
  <style>
    :root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --ink:#1d2430; --muted:#657083; --line:#d9dee7; --accent:#2463eb; --ok:#0f7b45; --bad:#b42318; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; background:#111827; color:white; }
    header h1 { margin:0; font-size:18px; font-weight:650; }
    header .sub { color:#cbd5e1; font-size:12px; }
    .top-controls { display:flex; align-items:flex-end; gap:12px; flex-wrap:wrap; justify-content:flex-end; }
    .topic-picker { min-width:220px; }
    .topic-picker label { color:#cbd5e1; }
    .topic-picker select { background:#1f2937; color:white; border-color:#334155; padding:7px 9px; }
    main { display:grid; grid-template-columns: 260px 1fr; min-height:calc(100vh - 54px); }
    nav { border-right:1px solid var(--line); background:#fff; padding:14px; }
    nav button { width:100%; display:flex; align-items:center; gap:8px; padding:10px 12px; margin:4px 0; border:0; border-radius:6px; background:transparent; color:var(--ink); text-align:left; cursor:pointer; }
    nav button.active, nav button:hover { background:#eaf0ff; color:#1743a2; }
    section { display:none; padding:18px; }
    section.active { display:block; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:14px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    .panel h2, .panel h3 { margin:0 0 10px; font-size:16px; }
    .row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    label { color:var(--muted); font-size:12px; }
    input, select, textarea { width:100%; border:1px solid var(--line); border-radius:6px; padding:9px 10px; background:white; color:var(--ink); font:inherit; }
    textarea { min-height:520px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size:12px; line-height:1.45; }
    button.primary, button.secondary, button.danger { border:0; border-radius:6px; padding:9px 12px; cursor:pointer; font-weight:600; }
    button.primary { background:var(--accent); color:white; }
    button.secondary { background:#e8edf5; color:#172033; }
    button.danger { background:#fee4e2; color:var(--bad); }
    .muted { color:var(--muted); }
    .ok { color:var(--ok); font-weight:650; }
    .bad { color:var(--bad); font-weight:650; }
    .status-card { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }
    .status-card.running { border-color:#86efac; background:#f0fdf4; }
    .status-card.stopped { border-color:#fecaca; background:#fef2f2; }
    .status-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }
    .status-title { font-size:18px; font-weight:750; }
    .status-dot { width:12px; height:12px; border-radius:999px; background:#94a3b8; box-shadow:0 0 0 4px rgba(148,163,184,.16); }
    .status-card.running .status-dot { background:var(--ok); box-shadow:0 0 0 4px rgba(15,123,69,.16); }
    .status-card.stopped .status-dot { background:var(--bad); box-shadow:0 0 0 4px rgba(180,35,24,.16); }
    .kv { display:grid; grid-template-columns:120px 1fr; gap:6px 10px; margin-top:8px; }
    .kv div:nth-child(odd) { color:var(--muted); }
    button:disabled { cursor:not-allowed; opacity:.52; }
    pre { margin:0; padding:12px; background:#0b1020; color:#dbeafe; border-radius:8px; overflow:auto; max-height:620px; white-space:pre-wrap; }
    .list { display:grid; gap:8px; }
    .item { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }
    .item-title { font-weight:650; margin-bottom:4px; }
    .pill { display:inline-flex; align-items:center; gap:4px; border:1px solid var(--line); border-radius:999px; padding:2px 8px; margin:2px; color:#344054; background:#f8fafc; font-size:12px; }
    .split { display:grid; grid-template-columns: 320px 1fr; gap:14px; }
    .markdown { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; max-height:720px; overflow:auto; }
    .markdown h1, .markdown h2, .markdown h3 { margin-top:1.1em; line-height:1.2; }
    .markdown h1:first-child, .markdown h2:first-child, .markdown h3:first-child { margin-top:0; }
    .markdown blockquote { margin:12px 0; padding:8px 12px; border-left:4px solid #b9c6da; background:#f8fafc; color:#344054; }
    .markdown code { background:#eef2f7; padding:1px 4px; border-radius:4px; }
    .markdown pre code { background:transparent; padding:0; }
    .markdown a { color:#1743a2; }
    .markdown table { border-collapse:collapse; width:100%; }
    .markdown th, .markdown td { border:1px solid var(--line); padding:6px 8px; }
    .preview-actions { margin-bottom:10px; display:flex; gap:8px; flex-wrap:wrap; }
    .wechat-frame { width:100%; min-height:720px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .config-tools { display:flex; align-items:flex-end; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
    .config-filter { min-width:260px; flex:1; }
    .config-table-wrap { border:1px solid var(--line); border-radius:8px; overflow:auto; max-height:720px; background:#fff; }
    .config-table { width:100%; border-collapse:collapse; table-layout:fixed; }
    .config-table th, .config-table td { border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }
    .config-table th { position:sticky; top:0; z-index:1; background:#f8fafc; text-align:left; color:#475467; font-size:12px; }
    .config-path { width:34%; overflow-wrap:anywhere; }
    .config-name { font-weight:650; }
    .config-path-hint { margin-top:2px; color:var(--muted); font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:11px; }
    .config-type { width:90px; color:var(--muted); font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:12px; }
    .config-value-cell { width:auto; }
    .config-value-cell input[type="text"], .config-value-cell input[type="number"], .config-value-cell textarea { font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:12px; }
    .config-value-cell textarea { min-height:80px; max-height:260px; resize:vertical; }
    .config-value-cell .prompt-area { min-height:180px; }
    .config-hidden { display:none; }
    .config-array-actions { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .config-add { border:0; border-radius:6px; padding:6px 10px; cursor:pointer; font-weight:650; background:#e8edf5; color:#172033; }
    @media (max-width: 860px) { main { grid-template-columns:1fr; } nav { display:flex; overflow:auto; border-right:0; border-bottom:1px solid var(--line); } nav button { min-width:150px; } .split { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>AI CTO Daily Console</h1><div class="sub">本地 Web UI · 运行流水线 · 查看日志 · 编辑配置 · 调试中间产物</div></div>
    <div class="top-controls">
      <div class="topic-picker">
        <label>Topic</label>
        <select id="topicSelect" onchange="changeTopic()"></select>
      </div>
      <div id="statusBadge" class="sub">Loading...</div>
    </div>
  </header>
  <main>
    <nav>
      <button class="active" data-tab="dashboard">Overview</button>
      <button data-tab="run">Run</button>
      <button data-tab="schedule">Schedule</button>
      <button data-tab="logs">Logs</button>
      <button data-tab="config">Config</button>
      <button data-tab="summaries">Summaries</button>
      <button data-tab="artifacts">Artifacts</button>
      <button data-tab="publish">Publish</button>
    </nav>
    <div>
      <section id="dashboard" class="active">
        <div class="grid">
          <div class="panel"><h2>Runtime</h2><div id="runtime"></div></div>
          <div class="panel"><h2>Scheduler</h2><div id="schedulerMini"></div></div>
          <div class="panel"><h2>Latest Jobs</h2><div id="jobsMini" class="list"></div></div>
          <div class="panel"><h2>Latest Summaries</h2><div id="summariesMini" class="list"></div></div>
        </div>
      </section>
      <section id="run">
        <div class="panel">
          <h2>Run Full AI CTO Daily Pipeline</h2>
          <div class="row">
            <div style="width:180px"><label>Hours</label><input id="runHours" type="number" min="1" value="24" /></div>
            <button class="primary" onclick="startRun()">Start Run</button>
            <button class="secondary" onclick="refreshAll()">Refresh</button>
          </div>
          <p class="muted">Runs the same entry point as `uv run ai-cto-daily --hours N`, with stdout/stderr captured to a log file.</p>
        </div>
      </section>
      <section id="schedule">
        <div class="panel">
          <h2>Scheduler</h2>
          <div id="scheduleStatus" class="list">Loading...</div>
          <div class="row" style="margin-top:12px">
            <div style="width:180px"><label>Run-now Hours</label><input id="scheduleRunHours" type="number" min="1" value="30" /></div>
            <button id="scheduleStartBtn" class="primary" onclick="startScheduler()">Start Scheduler</button>
            <button id="scheduleStopBtn" class="secondary" onclick="stopScheduler()">Stop Scheduler</button>
            <button class="secondary" onclick="runScheduleNow()">Run Now</button>
          </div>
          <p class="muted">Edit the schedule block in Config, save it, then start the scheduler. The standalone service command is `uv run ai-cto-daily-scheduler`.</p>
        </div>
      </section>
      <section id="logs">
        <div class="split">
          <div class="panel"><h2>Jobs</h2><div id="jobsList" class="list"></div></div>
          <div class="panel"><h2>Log</h2><pre id="logBox">Select a job.</pre></div>
        </div>
      </section>
      <section id="config">
        <div class="panel">
          <h2>Config Editor</h2>
          <div class="config-tools">
            <button class="secondary" onclick="loadConfig()">Reload</button>
            <button class="secondary" onclick="validateConfig()">Validate</button>
            <button class="primary" onclick="saveConfig()">Save with Backup</button>
            <button class="secondary" onclick="setConfigView('table')">Table View</button>
            <button class="secondary" onclick="setConfigView('raw')">Raw JSON</button>
            <div class="config-filter">
              <label>Filter table paths / values</label>
              <input id="configFilter" placeholder="topic, prompts.analysis, steam..." oninput="renderConfigTable()" />
            </div>
          </div>
          <div id="configTableWrap" class="config-table-wrap"></div>
          <textarea id="configText" class="config-hidden" spellcheck="false"></textarea>
          <p id="configMessage" class="muted"></p>
        </div>
      </section>
      <section id="summaries">
        <div class="split">
          <div class="panel"><h2>Files</h2><div id="summaryList" class="list"></div></div>
          <div>
            <div class="preview-actions">
              <button class="secondary" onclick="setSummaryMode('rendered')">Rendered</button>
              <button class="secondary" onclick="setSummaryMode('raw')">Raw Markdown</button>
            </div>
            <div class="markdown" id="summaryPreview">Select a summary.</div>
          </div>
        </div>
      </section>
      <section id="artifacts">
        <div class="split">
          <div class="panel"><h2>Runs</h2><div id="artifactRuns" class="list"></div></div>
          <div class="panel"><h2>Artifact Preview</h2><div class="markdown" id="artifactPreview">Select a run artifact.</div></div>
        </div>
      </section>
      <section id="publish">
        <div class="split">
          <div class="panel">
            <h2>WeChat Draft</h2>
            <p id="wechatStatus" class="muted">Loading...</p>
            <label>Mode</label>
            <select id="wechatMode">
              <option value="draft">Draft only</option>
              <option value="publish">Publish now</option>
            </select>
            <label>Summary</label>
            <select id="wechatSummary"></select>
            <label>Title</label>
            <input id="wechatTitle" maxlength="64" />
            <label>Digest</label>
            <textarea id="wechatDigest" style="min-height:80px" maxlength="120"></textarea>
            <label>Author</label>
            <input id="wechatAuthor" />
            <label>Cover Image</label>
            <input id="wechatCover" />
            <label>Original URL (optional)</label>
            <input id="wechatSourceUrl" placeholder="https://..." />
            <div class="row" style="margin-top:12px">
              <button class="secondary" onclick="previewWeChat()">Preview</button>
              <button class="primary" onclick="submitWeChat()">Submit</button>
            </div>
            <p id="wechatMessage" class="muted"></p>
          </div>
          <iframe class="wechat-frame" id="wechatPreview" title="WeChat preview"></iframe>
        </div>
      </section>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let selectedJob = null;
    let currentSummary = null;
    let summaryMode = 'rendered';
    let configView = 'table';
    let configObject = null;
    let selectedTopic = localStorage.getItem('aiCtoDailyTopic') || '';
    async function api(path, opts={}) {
      const res = await fetch(path, {headers:{'Content-Type':'application/json'}, ...opts});
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }
    function topicQuery() {
      return selectedTopic ? `topic=${encodeURIComponent(selectedTopic)}` : '';
    }
    function withTopic(path) {
      const q = topicQuery();
      if (!q) return path;
      return path + (path.includes('?') ? '&' : '?') + q;
    }
    function topicPayload(extra={}) {
      return selectedTopic ? {...extra, topic:selectedTopic} : extra;
    }
    function renderTopicSelect(topics, activeTopic) {
      const select = $('topicSelect');
      const previous = selectedTopic || activeTopic;
      select.innerHTML = (topics || []).map(t => `<option value="${esc(t.slug)}">${esc(t.name || t.slug)}${t.enabled ? '' : ' (disabled)'}</option>`).join('');
      if (previous && [...select.options].some(o => o.value === previous)) {
        select.value = previous;
        selectedTopic = previous;
      } else if (activeTopic) {
        select.value = activeTopic;
        selectedTopic = activeTopic;
      }
      if (selectedTopic) localStorage.setItem('aiCtoDailyTopic', selectedTopic);
    }
    async function changeTopic() {
      selectedTopic = $('topicSelect').value;
      localStorage.setItem('aiCtoDailyTopic', selectedTopic);
      currentSummary = null;
      $('summaryPreview').textContent = 'Select a summary.';
      $('artifactPreview').textContent = 'Select a run artifact.';
      $('wechatMode').dataset.loaded = '';
      await refreshAll();
    }
    function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
    document.querySelectorAll('nav button').forEach(btn => btn.onclick = () => {
      document.querySelectorAll('nav button, section').forEach(x => x.classList.remove('active'));
      btn.classList.add('active'); $(btn.dataset.tab).classList.add('active');
    });
    async function refreshAll() {
      const s = await api(withTopic('/api/status'));
      renderTopicSelect(s.topics || [], s.active_topic || ((s.validation || {}).topic || {}).slug);
      const v = s.validation;
      const scheduler = s.schedule || {};
      const schedulerLabel = scheduler.service_running ? 'Scheduler running' : 'Scheduler stopped';
      const schedulerClass = scheduler.service_running ? 'ok' : 'bad';
      $('statusBadge').innerHTML = `
        ${v.ok ? '<span class="ok">Config OK</span>' : '<span class="bad">Needs attention</span>'}
        &nbsp;|&nbsp;
        <span class="${schedulerClass}">${schedulerLabel}</span>
      `;
      $('runtime').innerHTML = v.error ? `<p class="bad">${esc(v.error)}</p>` : `
        <div><b>${esc(v.ai.provider)}</b> · ${esc(v.ai.model)}</div>
        <div><b>Topic</b> - ${esc((v.topic || {}).name || '')} <span class="muted">${esc((v.topic || {}).slug || '')}</span></div>
        <div class="muted">Languages: ${esc((v.ai.languages||[]).join(', '))}</div>
        <div class="muted">Threshold: ${esc(v.filtering.ai_score_threshold)}</div>
        <div class="muted">Artifacts: ${esc(v.artifacts.enabled)}</div>
        ${(v.missing_env||[]).map(x=>`<div class="bad">Missing env: ${esc(x)}</div>`).join('')}
        ${(v.warnings||[]).map(x=>`<div class="muted">${esc(x)}</div>`).join('')}
      `;
      $('schedulerMini').innerHTML = scheduleStatusHtml(scheduler, true);
      renderJobs(s.jobs || []);
      renderMini('jobsMini', (s.jobs||[]).map(j => ({title:j.id, meta:`${j.status} · ${j.created_at}`})));
      renderMini('summariesMini', (s.summaries||[]).map(f => ({title:f.name, meta:f.modified_at})));
      renderSummaries(s.summaries || []);
      renderArtifacts(s.artifact_runs || []);
      renderWeChatSummaries(s.summaries || []);
      renderSchedule(scheduler);
      refreshWeChatStatus();
    }
    function renderMini(id, items) {
      $(id).innerHTML = items.length ? items.map(x => `<div class="item"><div class="item-title">${esc(x.title)}</div><div class="muted">${esc(x.meta)}</div></div>`).join('') : '<p class="muted">None</p>';
    }
    function renderJobs(jobs) {
      $('jobsList').innerHTML = jobs.length ? jobs.map(j => `<div class="item">
        <div class="item-title">${esc(j.id)}</div>
        <div class="${j.status === 'success' ? 'ok' : j.status === 'error' ? 'bad' : 'muted'}">${esc(j.status)} ${j.returncode ?? ''}</div>
        <div class="muted">Topic: ${esc(j.topic || 'unknown')}</div>
        <button class="secondary" onclick="selectJob('${esc(j.id)}')">View Log</button>
      </div>`).join('') : '<p class="muted">No web jobs in this server session.</p>';
    }
    async function startRun() {
      const hours = Number($('runHours').value || 24);
      const job = await api('/api/run', {method:'POST', body:JSON.stringify(topicPayload({hours}))});
      selectedJob = job.id;
      await refreshAll();
      document.querySelector('[data-tab="logs"]').click();
      pollLog();
    }
    async function selectJob(id) { selectedJob = id; await pollLog(); }
    function renderSchedule(s) {
      $('scheduleStatus').innerHTML = scheduleStatusHtml(s, false);
      $('scheduleStartBtn').disabled = !!s.service_running;
      $('scheduleStopBtn').disabled = !s.service_running;
    }
    function scheduleStatusHtml(s, compact=false) {
      const cfg = s.config || {};
      const running = !!s.service_running;
      const enabled = !!s.enabled;
      const title = running ? 'Scheduler is running' : 'Scheduler is stopped';
      const nextRun = s.next_run_at || (enabled ? 'calculating...' : 'not scheduled');
      const mode = enabled ? 'Config enabled' : 'Config disabled';
      return `
        <div class="status-card ${running ? 'running' : 'stopped'}">
          <div class="status-head">
            <div>
              <div class="status-title">${title}</div>
              <div class="${enabled ? 'ok' : 'muted'}">${mode}</div>
            </div>
            <span class="status-dot" aria-hidden="true"></span>
          </div>
          <div class="kv">
            <div>Next run</div><div>${esc(nextRun)}</div>
            <div>Timezone</div><div>${esc(cfg.timezone || '')}</div>
            <div>Daily times</div><div>${esc((cfg.daily_times || []).join(', ') || 'none')}</div>
            <div>Interval</div><div>${esc(cfg.interval_minutes ?? 'none')}</div>
            <div>Pipeline hours</div><div>${esc(cfg.hours ?? '')}</div>
            ${compact ? '' : `<div>Current job</div><div>${s.current_job_running ? '<span class="ok">running</span>' : 'idle'}</div>`}
            <div>Last job</div><div>${esc(s.last_job_id || 'none')}</div>
            ${compact ? '' : `<div>Last log</div><div>${esc((s.last_job || {}).log_path || 'none')}</div>`}
          </div>
          ${s.last_error ? `<div class="bad" style="margin-top:10px">${esc(s.last_error)}</div>` : ''}
        </div>
      `;
    }
    async function startScheduler() {
      await api('/api/schedule/start', {method:'POST', body:JSON.stringify(topicPayload())});
      await refreshAll();
    }
    async function stopScheduler() {
      await api('/api/schedule/stop', {method:'POST', body:JSON.stringify(topicPayload())});
      await refreshAll();
    }
    async function runScheduleNow() {
      const hours = Number($('scheduleRunHours').value || 30);
      await api('/api/schedule/run-now', {method:'POST', body:JSON.stringify(topicPayload({hours}))});
      await refreshAll();
    }
    async function pollLog() {
      if (!selectedJob) return;
      const data = await api(`/api/jobs/${selectedJob}/log?tail=80000`);
      $('logBox').textContent = data.log || '(empty log)';
      if (data.job.status === 'running') setTimeout(pollLog, 2500);
    }
    async function loadConfig() {
      const data = await api('/api/config');
      $('configText').value = data.text;
      try { configObject = JSON.parse(data.text); }
      catch(e) { configObject = null; }
      setConfigView(configView || 'table');
      $('configMessage').textContent = data.path;
    }
    function setConfigView(mode) {
      configView = mode;
      if (mode === 'raw') {
        if (document.querySelector('[data-config-path]')) collectConfigFromTable();
        $('configText').classList.remove('config-hidden');
        $('configTableWrap').classList.add('config-hidden');
        $('configFilter').disabled = true;
        return;
      }
      try {
        configObject = JSON.parse($('configText').value || '{}');
        renderConfigTable();
        $('configText').classList.add('config-hidden');
        $('configTableWrap').classList.remove('config-hidden');
        $('configFilter').disabled = false;
      } catch(e) {
        $('configMessage').innerHTML = `<span class="bad">Cannot render table: ${esc(e.message)}</span>`;
        setConfigView('raw');
      }
    }
    function pathLabel(path) {
      return path.map((part, index) => typeof part === 'number' ? `[${part}]` : (index === 0 ? part : `.${part}`)).join('');
    }
    const CONFIG_LABELS = {
      version: 'Config version',
      active_topic: 'Active topic',
      topics: 'Topics',
      ai: 'AI model',
      provider: 'Provider',
      model: 'Model',
      api_key_env: 'API key env',
      base_url: 'Base URL',
      temperature: 'Temperature',
      max_tokens: 'Max tokens',
      throttle_sec: 'Throttle seconds',
      analysis_concurrency: 'Analysis concurrency',
      enrichment_concurrency: 'Enrichment concurrency',
      languages: 'Languages',
      email: 'Email',
      enabled: 'Enabled',
      smtp_server: 'SMTP server',
      smtp_port: 'SMTP port',
      smtp_username: 'SMTP username',
      imap_enabled: 'IMAP enabled',
      imap_server: 'IMAP server',
      imap_port: 'IMAP port',
      email_address: 'Email address',
      password_env: 'Password env',
      sender_name: 'Sender name',
      sources: 'Sources',
      github: 'GitHub sources',
      hackernews: 'Hacker News',
      rss: 'RSS feeds',
      reddit: 'Reddit',
      subreddits: 'Subreddits',
      users: 'Users',
      telegram: 'Telegram',
      channels: 'Channels',
      twitter: 'Twitter/X',
      steam: 'Steam',
      fetch_new_releases: 'Fetch new releases',
      candidate_count: 'Steam candidate count',
      new_release_window_days: 'New release window days',
      country: 'Country',
      language: 'Language',
      min_total_reviews: 'Minimum total reviews',
      min_positive_ratio: 'Minimum positive ratio',
      include_free: 'Include free games',
      include_early_access: 'Include Early Access',
      openbb: 'OpenBB',
      ossinsight: 'OSS Insight',
      name: 'Name',
      slug: 'Slug',
      title_en: 'English title',
      title_zh: 'Chinese title',
      description: 'Description',
      audience: 'Audience',
      relevance_prompt: 'Relevance prompt',
      cto_prompt_focus: 'Takeaway focus',
      prompts: 'Prompts',
      analysis_system: 'Scoring system prompt',
      analysis_user: 'Scoring user prompt',
      concept_system: 'Concept extraction system prompt',
      concept_user: 'Concept extraction user prompt',
      enrichment_system: 'Enrichment system prompt',
      enrichment_user: 'Enrichment user prompt',
      takeaway_system: 'Takeaway system prompt',
      takeaway_user: 'Takeaway user prompt',
      topic_dedup_system: 'Topic dedup system prompt',
      topic_dedup_user: 'Topic dedup user prompt',
      translation_system: 'Translation fallback system prompt',
      translation_user: 'Translation fallback user prompt',
      filtering: 'Filtering',
      ai_score_threshold: 'AI score threshold',
      time_window_hours: 'Time window hours',
      summary: 'Summary format',
      disclosure: 'Disclosure',
      include_summary: 'Opening summary',
      include_cto_takeaway: 'Takeaway section',
      max_detailed_items: 'Detailed item limit',
      show_scores: 'Show scores',
      compact_remaining: 'Compact remaining items',
      compact_sentence_limit: 'Compact sentence limit',
      include_bilingual: 'Bilingual output',
      bilingual_primary_language: 'Bilingual primary language',
      bilingual_secondary_language: 'Second language',
      cto_takeaway_ai_enabled: 'AI takeaway enabled',
      cto_takeaway_ai_items: 'AI takeaway item count',
      schedule: 'Schedule',
      timezone: 'Timezone',
      daily_times: 'Daily times',
      interval_minutes: 'Interval minutes',
      hours: 'Pipeline hours',
      run_on_start: 'Run on start',
      prevent_overlap: 'Prevent overlap',
      publishing: 'Publishing',
      wechat: 'WeChat',
      auto_create_draft_on_run: 'Auto-create draft after run',
      appid_env: 'AppID env',
      secret_env: 'Secret env',
      author: 'Author',
      cover_image: 'Cover image',
      default_digest: 'Default digest',
      publish_mode: 'Publish mode',
      review_reminder_enabled: 'Review reminder',
      review_reminder_recipients: 'Review reminder recipients',
      review_reminder_subject: 'Review reminder subject',
      review_url: 'Review URL',
      url: 'URL',
      category: 'Category',
      subreddit: 'Subreddit',
      fetch_limit: 'Fetch limit',
      min_score: 'Minimum score',
      fetch_comments: 'Fetch comments',
      sort: 'Sort',
      time_filter: 'Time filter'
    };
    function collectionItemLabel(parent, item, index) {
      if (!item || typeof item !== 'object') return `Item ${index + 1}`;
      if (parent === 'topics') return item.name || item.slug || `Topic ${index + 1}`;
      if (parent === 'rss') return item.name || item.url || `RSS feed ${index + 1}`;
      if (parent === 'github') return item.repo ? `${item.owner || ''}/${item.repo}` : (item.username || `GitHub source ${index + 1}`);
      if (parent === 'subreddits') return item.subreddit ? `r/${item.subreddit}` : `Subreddit ${index + 1}`;
      if (parent === 'channels') return item.channel || `Channel ${index + 1}`;
      if (parent === 'review_reminder_recipients' || parent === 'daily_times' || parent === 'languages' || parent === 'users') return String(item || `Item ${index + 1}`);
      return item.name || item.slug || item.title || `Item ${index + 1}`;
    }
    function readablePath(path) {
      const parts = [];
      let cursor = configObject;
      for (let i = 0; i < path.length; i++) {
        const part = path[i];
        const parent = path[i - 1];
        if (typeof part === 'number') {
          parts.push(collectionItemLabel(parent, cursor ? cursor[part] : null, part));
          cursor = cursor ? cursor[part] : undefined;
          continue;
        }
        parts.push(CONFIG_LABELS[part] || part.replace(/_/g, ' '));
        cursor = cursor ? cursor[part] : undefined;
      }
      return parts.join(' > ');
    }
    function flattenConfig(value, path=[], rows=[]) {
      if (value === null || typeof value !== 'object') {
        rows.push({path, value, type:value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value});
        return rows;
      }
      if (Array.isArray(value)) {
        rows.push({path, value, type:'array', arrayContainer:true});
      }
      const keys = Array.isArray(value) ? value.map((_, i) => i) : Object.keys(value);
      if (keys.length === 0) {
        rows.push({path, value, type:Array.isArray(value) ? 'array' : 'object', empty:true});
        return rows;
      }
      for (const key of keys) flattenConfig(value[key], path.concat([key]), rows);
      return rows;
    }
    function renderConfigTable() {
      if (!configObject) return;
      if (document.querySelector('[data-config-path]')) collectConfigFromTable();
      const filter = ($('configFilter').value || '').toLowerCase();
      const rows = flattenConfig(configObject).filter(row => {
        const label = pathLabel(row.path).toLowerCase();
        const readable = readablePath(row.path).toLowerCase();
        const value = row.value === null ? 'null' : String(row.value).toLowerCase();
        return !filter || label.includes(filter) || readable.includes(filter) || value.includes(filter);
      });
      $('configTableWrap').innerHTML = `
        <table class="config-table">
          <thead><tr><th class="config-path">Name</th><th class="config-type">Type</th><th>Value</th></tr></thead>
          <tbody>${rows.map(configRowHtml).join('')}</tbody>
        </table>
      `;
    }
    function configRowHtml(row) {
      const encodedPath = esc(JSON.stringify(row.path));
      const rawPath = pathLabel(row.path);
      const label = esc(readablePath(row.path));
      const hint = esc(rawPath);
      const nameCell = `<div class="config-name" title="${hint}">${label}</div><div class="config-path-hint">${hint}</div>`;
      if (row.arrayContainer) {
        const count = Array.isArray(row.value) ? row.value.length : 0;
        return `<tr><td class="config-path">${nameCell}</td><td class="config-type">array</td><td class="config-value-cell"><div class="config-array-actions"><button class="config-add" onclick="addConfigArrayItem('${encodedPath}')">Add item</button><span class="muted">${count} item${count === 1 ? '' : 's'}</span></div></td></tr>`;
      }
      if (row.empty) {
        return `<tr><td class="config-path">${nameCell}</td><td class="config-type">${row.type}</td><td class="config-value-cell muted">${row.type === 'array' ? 'Use Add item above.' : 'Edit empty objects in Raw JSON.'}</td></tr>`;
      }
      const value = row.value;
      const type = row.type;
      let control = '';
      if (type === 'boolean') {
        control = `<input data-config-path="${encodedPath}" data-config-type="boolean" type="checkbox" ${value ? 'checked' : ''} />`;
      } else if (type === 'number') {
        control = `<input data-config-path="${encodedPath}" data-config-type="number" type="number" step="any" value="${esc(value)}" />`;
      } else if (type === 'null') {
        control = `<input data-config-path="${encodedPath}" data-config-type="null" type="text" value="" placeholder="null; use Raw JSON to change type" disabled />`;
      } else {
        const text = String(value ?? '');
        const longText = text.length > 100 || text.includes('\n') || rawPath.includes('.prompts.');
        control = longText
          ? `<textarea data-config-path="${encodedPath}" data-config-type="string" class="${rawPath.includes('.prompts.') ? 'prompt-area' : ''}">${esc(text)}</textarea>`
          : `<input data-config-path="${encodedPath}" data-config-type="string" type="text" value="${esc(text)}" />`;
      }
      return `<tr><td class="config-path">${nameCell}</td><td class="config-type">${type}</td><td class="config-value-cell">${control}</td></tr>`;
    }
    function setAtPath(obj, path, value) {
      let target = obj;
      for (let i = 0; i < path.length - 1; i++) target = target[path[i]];
      target[path[path.length - 1]] = value;
    }
    function getAtPath(obj, path) {
      let target = obj;
      for (const part of path) target = target ? target[part] : undefined;
      return target;
    }
    function emptyLike(value, path=[]) {
      if (Array.isArray(value)) return [];
      if (value && typeof value === 'object') {
        const out = {};
        for (const [key, child] of Object.entries(value)) out[key] = emptyLike(child, path.concat([key]));
        if ('enabled' in out) out.enabled = true;
        return out;
      }
      if (typeof value === 'boolean') return false;
      if (typeof value === 'number') return 0;
      return '';
    }
    function defaultArrayItem(path, arr) {
      const key = path[path.length - 1];
      if (key === 'daily_times') return '09:00';
      if (key === 'languages') return 'en';
      if (key === 'review_reminder_recipients') return 'you@example.com';
      if (key === 'users') return 'example';
      if (key === 'keywords') return '';
      if (key === 'symbols') return '';
      if (key === 'topics') return {
        slug: 'new-topic',
        name: 'New Topic',
        enabled: true,
        title_en: 'New Daily',
        title_zh: '',
        description: '',
        audience: '',
        relevance_prompt: '',
        cto_prompt_focus: '',
        prompts: {},
        sources: {github: [], hackernews: {enabled: false}, rss: [], reddit: {enabled: false, subreddits: [], users: [], fetch_comments: 0}, telegram: {enabled: false, channels: []}},
        filtering: {ai_score_threshold: 7, time_window_hours: 24},
        summary: {}
      };
      if (key === 'rss') return {name: 'New RSS feed', url: 'https://example.com/feed.xml', enabled: true, category: ''};
      if (key === 'github') return {type: 'repo_releases', owner: '', repo: '', enabled: true};
      if (key === 'subreddits') return {subreddit: 'example', enabled: true, sort: 'hot', time_filter: 'day', fetch_limit: 10, min_score: 0};
      if (key === 'channels') return {channel: 'example', enabled: true, fetch_limit: 20};
      if (key === 'watchlists') return {name: 'New Watchlist', symbols: [], enabled: true, provider: 'yfinance', fetch_limit: 20, category: ''};
      if (arr.length) return emptyLike(arr[arr.length - 1], path.concat([arr.length]));
      return '';
    }
    function addConfigArrayItem(encodedPath) {
      try {
        if (document.querySelector('[data-config-path]')) collectConfigFromTable();
        const path = JSON.parse(encodedPath);
        const arr = getAtPath(configObject, path);
        if (!Array.isArray(arr)) throw new Error('Target is not an array.');
        arr.push(defaultArrayItem(path, arr));
        $('configText').value = JSON.stringify(configObject, null, 2) + '\n';
        renderConfigTable();
        $('configMessage').innerHTML = '<span class="ok">Item added. Review it, then save.</span>';
      } catch(e) {
        $('configMessage').innerHTML = `<span class="bad">${esc(e.message)}</span>`;
      }
    }
    function collectConfigFromTable() {
      const obj = JSON.parse(JSON.stringify(configObject || {}));
      document.querySelectorAll('[data-config-path]').forEach(input => {
        const path = JSON.parse(input.dataset.configPath);
        const type = input.dataset.configType;
        let value = input.value;
        if (type === 'boolean') value = input.checked;
        else if (type === 'number') value = value === '' ? null : Number(value);
        else if (type === 'null') value = null;
        setAtPath(obj, path, value);
      });
      configObject = obj;
      $('configText').value = JSON.stringify(obj, null, 2) + '\n';
      return $('configText').value;
    }
    function currentConfigText() {
      return configView === 'table' ? collectConfigFromTable() : $('configText').value;
    }
    async function validateConfig() {
      try {
        await api('/api/config/validate', {method:'POST', body:JSON.stringify({text:currentConfigText()})});
        $('configMessage').innerHTML = '<span class="ok">Config is valid.</span>';
      } catch(e) { $('configMessage').innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
    }
    async function saveConfig() {
      try {
        const data = await api('/api/config', {method:'POST', body:JSON.stringify({text:currentConfigText()})});
        $('configMessage').innerHTML = `<span class="ok">Saved.</span> Backup: ${esc(data.backup)}`;
        if (configView === 'table') renderConfigTable();
        await refreshAll();
      } catch(e) { $('configMessage').innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
    }
    function renderSummaries(items) {
      $('summaryList').innerHTML = items.length ? items.map(f => `<div class="item">
        <div class="item-title">${esc(f.name)}</div><div class="muted">${esc(f.modified_at)}</div>
        <button class="secondary" onclick="openSummary('${esc(f.name)}')">Open</button>
      </div>`).join('') : '<p class="muted">No summaries.</p>';
    }
    function renderWeChatSummaries(items) {
      const select = $('wechatSummary');
      const selected = select.value;
      select.innerHTML = items.map(f => `<option value="${esc(f.name)}">${esc(f.name)}</option>`).join('');
      if (selected) select.value = selected;
    }
    async function refreshWeChatStatus() {
      try {
        const s = await api(withTopic('/api/publish/wechat/status'));
        $('wechatStatus').innerHTML = `
          <span class="${s.enabled && s.appid_present && s.secret_present && s.cover_exists ? 'ok' : 'bad'}">
            ${s.enabled ? 'Enabled' : 'Disabled'}
          </span>
          · AppID ${s.appid_present ? 'OK' : 'missing'}
          · Secret ${s.secret_present ? 'OK' : 'missing'}
          · Cover ${s.cover_exists ? 'OK' : 'missing'}
        `;
        if (!$('wechatAuthor').value) $('wechatAuthor').value = s.author || '';
        if (!$('wechatCover').value) $('wechatCover').value = s.cover_image || '';
        if (!$('wechatMode').dataset.loaded) {
          $('wechatMode').value = s.publish_mode || 'draft';
          $('wechatMode').dataset.loaded = '1';
        }
      } catch(e) {
        $('wechatStatus').innerHTML = `<span class="bad">${esc(e.message)}</span>`;
      }
    }
    async function previewWeChat() {
      try {
        const payload = {
          topic: selectedTopic,
          summary_name: $('wechatSummary').value,
          title: $('wechatTitle').value,
          digest: $('wechatDigest').value,
          author: $('wechatAuthor').value,
          content_source_url: $('wechatSourceUrl').value
        };
        const data = await api('/api/publish/wechat/preview', {method:'POST', body:JSON.stringify(payload)});
        $('wechatTitle').value = data.title || '';
        $('wechatDigest').value = data.digest || '';
        $('wechatAuthor').value = data.author || '';
        $('wechatPreview').srcdoc = `<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;padding:18px;background:#fff;}</style></head><body>${data.html}</body></html>`;
        $('wechatMessage').innerHTML = '<span class="ok">Preview rendered.</span>';
      } catch(e) {
        $('wechatMessage').innerHTML = `<span class="bad">${esc(e.message)}</span>`;
      }
    }
    async function submitWeChat() {
      const mode = $('wechatMode').value || 'draft';
      const actionText = mode === 'publish' ? 'publish this article now' : 'create a WeChat draft';
      if (!confirm(`Really ${actionText}?`)) return;
      try {
        const payload = {
          mode,
          topic: selectedTopic,
          summary_name: $('wechatSummary').value,
          title: $('wechatTitle').value,
          digest: $('wechatDigest').value,
          author: $('wechatAuthor').value,
          cover_image: $('wechatCover').value,
          content_source_url: $('wechatSourceUrl').value
        };
        $('wechatMessage').textContent = mode === 'publish'
          ? 'Creating draft and submitting to WeChat publishing...'
          : 'Publishing to WeChat draft box...';
        const data = await api('/api/publish/wechat/submit', {method:'POST', body:JSON.stringify(payload)});
        const done = data.mode === 'publish' ? 'Publish submitted.' : 'Draft created.';
        $('wechatMessage').innerHTML = `<span class="ok">${done}</span> media_id: ${esc(data.media_id)}${data.publish_id ? `<br>publish_id: ${esc(data.publish_id)}` : ''}<br>record: ${esc(data.record_path)}`;
      } catch(e) {
        $('wechatMessage').innerHTML = `<span class="bad">${esc(e.message)}</span>`;
      }
    }
    async function openSummary(name) {
      const data = await api(`/api/summaries/${encodeURIComponent(name)}`);
      currentSummary = data;
      renderSummaryPreview();
    }
    function setSummaryMode(mode) {
      summaryMode = mode;
      renderSummaryPreview();
    }
    function renderSummaryPreview() {
      if (!currentSummary) return;
      if (summaryMode === 'raw') {
        $('summaryPreview').innerHTML = `<pre style="background:white;color:#1d2430;max-height:none">${esc(currentSummary.text)}</pre>`;
      } else {
        $('summaryPreview').innerHTML = currentSummary.html || `<pre style="background:white;color:#1d2430;max-height:none">${esc(currentSummary.text)}</pre>`;
      }
    }
    function renderArtifacts(runs) {
      $('artifactRuns').innerHTML = runs.length ? runs.map(r => `<div class="item">
        <div class="item-title">${esc(r.id)}</div><div class="muted">${esc(r.modified_at)}</div>
        <div>${(r.stages||[]).map(s=>`<button class="secondary" onclick="openStage('${esc(r.id)}','${esc(s)}')">${esc(s)}</button>`).join(' ')}</div>
        <div style="margin-top:6px">${(r.summaries||[]).map(s=>`<button class="secondary" onclick="openRunSummary('${esc(r.id)}','${esc(s)}')">${esc(s)}</button>`).join(' ')}
        ${r.has_ai_calls ? `<button class="secondary" onclick="openAiCalls('${esc(r.id)}')">AI Calls</button>` : ''}</div>
      </div>`).join('') : '<p class="muted">No artifact runs.</p>';
    }
    async function openStage(run, stage) {
      const data = await api(`/api/artifact-runs/${encodeURIComponent(run)}/stage/${encodeURIComponent(stage)}`);
      $('artifactPreview').innerHTML = `<pre>${esc(JSON.stringify(data.items.slice(0, 50), null, 2))}</pre>`;
    }
    async function openRunSummary(run, name) {
      const data = await api(`/api/artifact-runs/${encodeURIComponent(run)}/summary/${encodeURIComponent(name)}`);
      $('artifactPreview').innerHTML = data.html || esc(data.text);
    }
    async function openAiCalls(run) {
      const data = await api(`/api/artifact-runs/${encodeURIComponent(run)}/ai-calls`);
      $('artifactPreview').innerHTML = `<pre>${esc(JSON.stringify(data.items, null, 2))}</pre>`;
    }
    loadConfig(); refreshAll(); setInterval(refreshAll, 10000);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="AI CTO Daily local web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Do not auto-start the scheduler even when schedule.enabled is true.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env", override=False)

    server = ThreadingHTTPServer((args.host, args.port), AICTODailyWebHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"AI CTO Daily web console running at {url}")
    if not args.no_scheduler:
        try:
            scheduler = _get_scheduler()
            if _load_schedule_config().enabled:
                scheduler.start()
                print("AI CTO Daily scheduler auto-started from config.")
        except Exception as exc:
            print(f"Scheduler was not started: {exc}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for scheduler in list(SCHEDULERS.values()):
            scheduler.stop()
        server.server_close()


if __name__ == "__main__":
    main()
