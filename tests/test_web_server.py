"""Tests for the local web console helpers."""

import json

import pytest

from src.storage.manager import ConfigError
from src.web.server import _render_markdown, _validate_config_text
from src.publishers.wechat_renderer import append_full_version_note, markdown_to_wechat_html


def test_validate_config_text_accepts_example_config():
    with open("data/config.example.json", "r", encoding="utf-8") as f:
        config = _validate_config_text(f.read())

    assert config.ai.model
    assert config.filtering.ai_score_threshold > 0


def test_validate_config_text_accepts_wechat_publish_mode():
    with open("data/config.example.json", "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    data["publishing"]["wechat"]["publish_mode"] = "publish"

    config = _validate_config_text(json.dumps(data))

    assert config.publishing.wechat.publish_mode == "publish"


def test_validate_config_text_rejects_invalid_json():
    with pytest.raises(ConfigError):
        _validate_config_text("{not-json")


def test_validate_config_text_rejects_invalid_schema():
    with pytest.raises(ConfigError):
        _validate_config_text(json.dumps({"ai": {"provider": "openai"}}))


def test_render_markdown_outputs_html():
    html = _render_markdown("# Title\n\n- one\n- two")

    assert "<h1" in html
    assert "<li>one</li>" in html


def test_wechat_full_version_note_has_no_plain_full_url():
    html = markdown_to_wechat_html(
        append_full_version_note("# Title", "https://example.com/full")
    )

    assert "完整链接版" in html
    assert "https://example.com/full" not in html
