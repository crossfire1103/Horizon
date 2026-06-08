"""WeChat Official Account draft publishing."""

from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..models import WeChatPublishingConfig
from .wechat_renderer import (
    append_full_version_note,
    extract_digest,
    extract_title,
    markdown_to_wechat_html,
)


WECHAT_API_BASE = "https://api.weixin.qq.com"


class WeChatPublishError(RuntimeError):
    """Raised when WeChat publishing fails."""


@dataclass
class WeChatDraftResult:
    media_id: str
    title: str
    digest: str
    thumb_media_id: str
    record_path: str | None = None


class WeChatPublisher:
    def __init__(
        self,
        config: WeChatPublishingConfig,
        *,
        root_dir: Path | str = ".",
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.root_dir = Path(root_dir).resolve()
        self.client = client
        self._own_client = client is None

    async def __aenter__(self) -> "WeChatPublisher":
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self.client is not None and self._own_client:
            await self.client.aclose()

    async def publish_markdown_draft(
        self,
        markdown_text: str,
        *,
        title: str | None = None,
        digest: str | None = None,
        author: str | None = None,
        cover_image: str | None = None,
        content_source_url: str | None = None,
        record_dir: Path | None = None,
    ) -> WeChatDraftResult:
        if self.client is None:
            raise RuntimeError("Use WeChatPublisher as an async context manager.")

        final_title = (title or extract_title(markdown_text)).strip()[:64]
        final_digest = (digest or extract_digest(markdown_text, self.config.default_digest)).strip()[:120]
        final_author = (author or self.config.author).strip()
        wechat_markdown = append_full_version_note(markdown_text, content_source_url)
        html = markdown_to_wechat_html(wechat_markdown)

        token = await self.get_access_token()
        thumb_media_id = await self.upload_thumb(token, cover_image or self.config.cover_image)
        media_id = await self.add_draft(
            token,
            title=final_title,
            author=final_author,
            digest=final_digest,
            content=html,
            thumb_media_id=thumb_media_id,
            content_source_url=content_source_url,
        )

        result = WeChatDraftResult(
            media_id=media_id,
            title=final_title,
            digest=final_digest,
            thumb_media_id=thumb_media_id,
        )
        if record_dir:
            result.record_path = str(self.save_record(record_dir, result))
        return result

    async def get_access_token(self) -> str:
        appid = self._env_value(self.config.appid_env)
        secret = self._env_value(self.config.secret_env)
        response = await self.client.get(
            f"{WECHAT_API_BASE}/cgi-bin/token",
            params={
                "grant_type": "client_credential",
                "appid": appid,
                "secret": secret,
            },
        )
        payload = self._json_response(response)
        token = payload.get("access_token")
        if not token:
            raise WeChatPublishError(f"WeChat access_token missing: {payload}")
        return str(token)

    async def upload_thumb(self, access_token: str, cover_image: str) -> str:
        cover_path = self._resolve_path(cover_image)
        if not cover_path.exists():
            raise WeChatPublishError(f"WeChat cover image not found: {cover_path}")
        media_type = mimetypes.guess_type(str(cover_path))[0] or "application/octet-stream"
        with cover_path.open("rb") as f:
            response = await self.client.post(
                f"{WECHAT_API_BASE}/cgi-bin/material/add_material",
                params={"access_token": access_token, "type": "thumb"},
                files={"media": (cover_path.name, f, media_type)},
            )
        payload = self._json_response(response)
        media_id = payload.get("media_id")
        if not media_id:
            raise WeChatPublishError(f"WeChat thumb media_id missing: {payload}")
        return str(media_id)

    async def add_draft(
        self,
        access_token: str,
        *,
        title: str,
        author: str,
        digest: str,
        content: str,
        thumb_media_id: str,
        content_source_url: str | None = None,
    ) -> str:
        article = {
            "title": title,
            "author": author,
            "digest": digest,
            "content": content,
            "thumb_media_id": thumb_media_id,
            "need_open_comment": 0,
            "only_fans_can_comment": 0,
        }
        if content_source_url:
            article["content_source_url"] = content_source_url

        response = await self.client.post(
            f"{WECHAT_API_BASE}/cgi-bin/draft/add",
            params={"access_token": access_token},
            json={"articles": [article]},
        )
        payload = self._json_response(response)
        media_id = payload.get("media_id")
        if not media_id:
            raise WeChatPublishError(f"WeChat draft media_id missing: {payload}")
        return str(media_id)

    def save_record(self, record_dir: Path, result: WeChatDraftResult) -> Path:
        record_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = record_dir / f"{timestamp}-{result.media_id}.json"
        path.write_text(
            json.dumps(
                {
                    "media_id": result.media_id,
                    "title": result.title,
                    "digest": result.digest,
                    "thumb_media_id": result.thumb_media_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root_dir / path
        return path.resolve()

    @staticmethod
    def _json_response(response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise WeChatPublishError(f"WeChat HTTP error: {exc}") from exc
        errcode = payload.get("errcode")
        if errcode not in (None, 0):
            raise WeChatPublishError(f"WeChat API error {errcode}: {payload.get('errmsg')}")
        return payload

    @staticmethod
    def _env_value(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise WeChatPublishError(f"Missing environment variable: {name}")
        return value
