"""Tests for WeChat publishing helpers."""

import asyncio
import httpx

from src.models import WeChatPublishingConfig
from src.publishers.wechat import WeChatPublisher
from src.publishers.wechat_renderer import (
    append_full_version_note,
    extract_digest,
    extract_title,
    markdown_to_wechat_html,
)


def test_wechat_renderer_outputs_inline_html():
    html = markdown_to_wechat_html("# AI CTO Daily\n\n> intro\n\n- one")

    assert html.startswith("<section")
    assert "AI CTO Daily" in html
    assert "style=" in html
    assert "• one" in html
    assert "<li" not in html


def test_wechat_renderer_strips_external_links_from_body():
    html = markdown_to_wechat_html("[OpenAI](https://openai.com) and [local](#item-1)")

    assert "OpenAI" in html
    assert "local" in html
    assert "https://openai.com" not in html
    assert "href=" not in html


def test_wechat_renderer_keeps_original_links_as_plain_urls():
    html = markdown_to_wechat_html("[原文](https://example.com/original)")

    assert "原文" in html
    assert "链接：" in html
    assert "https://example.com/original" in html
    assert "href=" not in html


def test_wechat_renderer_uses_english_link_label_for_original():
    html = markdown_to_wechat_html("[Original](https://example.com/original)")

    assert "Link: https://example.com/original" in html
    assert "链接：" not in html


def test_wechat_renderer_removes_empty_list_items():
    html = markdown_to_wechat_html("- one\n- \n- two")

    assert "one" in html
    assert "two" in html
    assert "<li" not in html


def test_wechat_renderer_flattens_ordered_lists():
    html = markdown_to_wechat_html("1. first\n2. second")

    assert "1. first" in html
    assert "2. second" in html
    assert "<ol" not in html


def test_wechat_renderer_drops_details_reference_blocks():
    html = markdown_to_wechat_html(
        "<details><summary>References</summary><ul>"
        "<li><a href=\"https://example.com/ref\">Ref</a></li>"
        "</ul></details>"
    )

    assert "References" not in html
    assert "https://example.com/ref" not in html


def test_wechat_renderer_does_not_append_heading_links_to_end():
    html = markdown_to_wechat_html("## [Item Title](https://example.com/item)\n\nBody")

    assert "Item Title" in html
    assert "https://example.com/item" not in html


def test_append_full_version_note_adds_read_original_hint():
    text = append_full_version_note("# Title", "https://example.com/full")

    assert "完整链接版请点击文末" in text
    assert "https://example.com/full" not in text


def test_extract_title_and_digest():
    text = "# AI CTO Daily - 2026-06-08\n\n> From 10 items, 3 selected"

    assert extract_title(text) == "AI CTO Daily - 2026-06-08"
    assert extract_digest(text, "fallback") == "From 10 items, 3 selected"


def test_extract_title_normalizes_legacy_horizon_title():
    assert extract_title("# Horizon Daily - 2026-06-08") == "AI CTO Daily - 2026-06-08"


def test_wechat_publisher_creates_draft(tmp_path, monkeypatch):
    cover = tmp_path / "cover.png"
    cover.write_bytes(b"fake-image")
    monkeypatch.setenv("WECHAT_APP_ID", "appid")
    monkeypatch.setenv("WECHAT_APP_SECRET", "secret")

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/cgi-bin/token":
            assert request.url.params["appid"] == "appid"
            assert request.url.params["secret"] == "secret"
            return httpx.Response(200, json={"access_token": "token"})
        if request.url.path == "/cgi-bin/material/add_material":
            assert request.url.params["access_token"] == "token"
            assert request.url.params["type"] == "thumb"
            return httpx.Response(200, json={"media_id": "thumb123"})
        if request.url.path == "/cgi-bin/draft/add":
            payload = __import__("json").loads(request.content.decode("utf-8"))
            article = payload["articles"][0]
            assert article["thumb_media_id"] == "thumb123"
            assert article["title"] == "My Title"
            assert "content" in article
            return httpx.Response(200, json={"media_id": "draft123"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            publisher = WeChatPublisher(
                WeChatPublishingConfig(cover_image=str(cover)),
                root_dir=tmp_path,
                client=client,
            )
            return await publisher.publish_markdown_draft(
                "# My Title\n\nhello",
                record_dir=tmp_path / "records",
            )

    result = asyncio.run(run())

    assert result.media_id == "draft123"
    assert result.thumb_media_id == "thumb123"
    assert result.record_path
    assert len(seen) == 3
