import asyncio
from datetime import datetime, timezone

from src.ai.analyzer import ContentAnalyzer
from src.ai.utils import detect_original_language
from src.models import ContentItem, SourceType


class RecordingClient:
    def __init__(self):
        self.user_prompt = ""

    async def complete(self, system, user, temperature=None, max_tokens=None):
        self.user_prompt = user
        return '{"score": 8, "reason": "ok", "summary": "保留原文语义。", "tags": ["ai"]}'


def test_detect_original_language_distinguishes_cjk_and_english():
    assert detect_original_language("这是一个中文标题，关于人工智能模型。") == "zh"
    assert detect_original_language("This is an English title about AI systems.") == "en"
    assert detect_original_language("") == "unknown"


def test_analyzer_records_original_language_and_includes_it_in_prompt():
    client = RecordingClient()
    analyzer = ContentAnalyzer(client)
    item = ContentItem(
        id="rss:item-1",
        source_type=SourceType.RSS,
        title="中文模型发布：新的推理能力",
        url="https://example.com/news",
        content="这是一篇中文文章，介绍新的模型能力和部署方式。",
        author="tester",
        published_at=datetime(2026, 6, 8, tzinfo=timezone.utc),
    )

    asyncio.run(analyzer._analyze_item(item))

    assert item.metadata["original_language"] == "zh"
    assert "Original language: zh" in client.user_prompt
    assert item.ai_summary == "保留原文语义。"
