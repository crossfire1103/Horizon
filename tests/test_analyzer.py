import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import src.ai.analyzer as analyzer_module
from src.ai.analyzer import ContentAnalyzer
from src.models import ContentItem, PromptConfig, SourceType, TopicConfig


def _make_item(item_id: str) -> ContentItem:
    return ContentItem(
        id=item_id,
        source_type=SourceType.RSS,
        title=f"Item {item_id}",
        url="https://example.com/item",
        published_at=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )


def test_analyze_batch_does_not_sleep_by_default(monkeypatch):
    analyzer = ContentAnalyzer(SimpleNamespace())
    items = [_make_item("rss:test:1"), _make_item("rss:test:2")]
    sleep_calls = []

    async def fake_analyze_item(item):
        item.ai_score = 8.0

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)
    monkeypatch.setattr(analyzer_module.asyncio, "sleep", fake_sleep)

    result = asyncio.run(analyzer.analyze_batch(items))

    assert len(result) == 2
    assert sleep_calls == []


def test_analyze_batch_sleeps_between_items_when_throttle_configured(monkeypatch):
    client = SimpleNamespace(config=SimpleNamespace(throttle_sec=1.5))
    analyzer = ContentAnalyzer(client)
    items = [_make_item("rss:test:1"), _make_item("rss:test:2"), _make_item("rss:test:3")]
    sleep_calls = []

    async def fake_analyze_item(item):
        item.ai_score = 8.0

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)
    monkeypatch.setattr(analyzer_module.asyncio, "sleep", fake_sleep)

    asyncio.run(analyzer.analyze_batch(items))

    assert sleep_calls == [1.5, 1.5]


def test_analyze_batch_concurrent_processing(monkeypatch):
    """Verify that higher concurrency allows overlapping item processing."""
    client = SimpleNamespace(config=SimpleNamespace(analysis_concurrency=3))
    analyzer = ContentAnalyzer(client)
    items = [_make_item(f"rss:test:{i}") for i in range(5)]
    active_count = 0
    max_active = 0

    async def fake_analyze_item(item):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.05)  # Small delay to allow overlap
        active_count -= 1

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)

    asyncio.run(analyzer.analyze_batch(items))

    assert max_active == 3
    assert all(item.ai_score is None for item in items)  # None because fake_analyze_item doesn't set it


def test_analyze_batch_concurrent_preserves_order(monkeypatch):
    """Verify that analyze_batch preserves input order in results."""
    client = SimpleNamespace(config=SimpleNamespace(analysis_concurrency=3))
    analyzer = ContentAnalyzer(client)
    items = [_make_item(f"rss:test:{i}") for i in range(5)]

    async def fake_analyze_item(item):
        item.ai_score = float(item.id.split(":")[-1]) * 10

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)

    result = asyncio.run(analyzer.analyze_batch(items))

    assert [item.id for item in result] == [item.id for item in items]


def test_analyze_item_includes_feed_and_category_in_source_prompt():
    class FakeClient:
        config = SimpleNamespace()

        def __init__(self):
            self.user_prompt = ""

        async def complete(self, system, user):
            self.user_prompt = user
            return '{"score": 8, "reason": "Relevant", "summary": "Summary", "tags": ["games"]}'

    client = FakeClient()
    analyzer = ContentAnalyzer(client)
    item = _make_item("rss:test:source")
    item.metadata["feed_name"] = "Steam News Hub"
    item.metadata["category"] = "steam"

    asyncio.run(analyzer._analyze_item(item))

    assert "Source: rss | feed=Steam News Hub | category=steam" in client.user_prompt
    assert item.ai_score == 8.0


def test_analyze_item_uses_topic_prompt_overrides():
    class FakeClient:
        config = SimpleNamespace()

        def __init__(self):
            self.system_prompt = ""
            self.user_prompt = ""

        async def complete(self, system, user):
            self.system_prompt = system
            self.user_prompt = user
            return '{"score": 7, "reason": "Custom", "summary": "Summary", "tags": ["custom"]}'

    topic = TopicConfig(
        slug="gaming",
        name="Gaming",
        prompts=PromptConfig(
            analysis_system="Custom system prompt",
            analysis_user="Custom user prompt for {title} from {source} at {published_at}",
        ),
    )
    client = FakeClient()
    analyzer = ContentAnalyzer(client, topic=topic)

    asyncio.run(analyzer._analyze_item(_make_item("rss:test:prompt")))

    assert client.system_prompt == "Custom system prompt"
    assert (
        client.user_prompt
        == "Custom user prompt for Item rss:test:prompt from rss at 2026-04-26T00:00:00+00:00"
    )
