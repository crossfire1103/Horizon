"""Filesystem artifacts and AI call tracing."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from ..models import ArtifactConfig


_AI_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("ai_trace_context", default={})


@contextmanager
def ai_trace_context(**metadata: Any) -> Iterator[None]:
    """Attach stage/item metadata to AI calls made inside the context."""
    current = dict(_AI_TRACE_CONTEXT.get())
    current.update({k: v for k, v in metadata.items() if v is not None})
    token = _AI_TRACE_CONTEXT.set(current)
    try:
        yield
    finally:
        _AI_TRACE_CONTEXT.reset(token)


@dataclass
class ArtifactStore:
    """Small file-backed run store for CLI debugging and replay."""

    root: Path
    run_id: str
    cache_ai_calls: bool = False
    replay_ai_calls: bool = False

    @classmethod
    def from_config(cls, config: ArtifactConfig) -> "ArtifactStore | None":
        if not config.enabled:
            return None
        root = Path(config.root_dir)
        run_id = cls._make_run_id()
        store = cls(
            root=root,
            run_id=run_id,
            cache_ai_calls=config.cache_ai_calls,
            replay_ai_calls=config.replay_ai_calls,
        )
        store.run_dir.mkdir(parents=True, exist_ok=True)
        store.cache_dir.mkdir(parents=True, exist_ok=True)
        store.write_json(
            "meta.json",
            {
                "run_id": run_id,
                "created_at": cls._utc_now(),
                "cache_ai_calls": config.cache_ai_calls,
                "replay_ai_calls": config.replay_ai_calls,
            },
        )
        return store

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def cache_dir(self) -> Path:
        return self.root / "_ai-cache"

    def write_json(self, filename: str, payload: Any) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_text(self, filename: str, text: str) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def append_jsonl(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def save_stage(self, stage: str, payload: Any) -> Path:
        return self.write_json(f"stages/{stage}.json", payload)

    def save_summary(self, language: str, markdown: str) -> Path:
        return self.write_text(f"summaries/summary-{language}.md", markdown)

    def ai_cache_key(
        self,
        *,
        provider: str,
        model: str,
        system: str,
        user: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load_ai_cache(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        response = payload.get("response")
        return response if isinstance(response, str) else None

    def save_ai_cache(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def record_ai_call(self, payload: dict[str, Any]) -> Path:
        record = {
            "timestamp": self._utc_now(),
            "run_id": self.run_id,
            "context": _AI_TRACE_CONTEXT.get(),
            **payload,
        }
        return self.append_jsonl("ai_calls.jsonl", record)

    @staticmethod
    def _make_run_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run-{now}-{uuid4().hex[:8]}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

