"""Unit tests for daily summary rendering."""

import asyncio
from datetime import datetime, timezone

from src.ai.summarizer import DailySummarizer
from src.models import ContentItem, SourceType, SummaryConfig, TopicConfig


def _run_async(coro):
    return asyncio.run(coro)


def _make_item(idx: int) -> ContentItem:
    item = ContentItem(
        id=f"rss:item-{idx}",
        source_type=SourceType.RSS,
        title=f"Important Item {idx}",
        url=f"https://example.com/items/{idx}",
        content="content",
        author="tester",
        published_at=datetime(2026, 4, 25, 8, 0, tzinfo=timezone.utc),
    )
    item.ai_score = 8.0
    item.ai_summary = f"Summary for item {idx}."
    item.ai_tags = ["AI", "News"]
    return item


def test_generate_webhook_overview_lists_items_without_full_details():
    summarizer = DailySummarizer()
    items = [_make_item(1), _make_item(2)]

    result = summarizer.generate_webhook_overview(
        items,
        date="2026-04-25",
        total_fetched=10,
        language="en",
    )

    assert "Selected 2 important items from 10 fetched items" in result
    assert "1. [Important Item 1](https://example.com/items/1)" in result
    assert "2. [Important Item 2](https://example.com/items/2)" in result
    assert "Summary for item 1." not in result


def test_generate_summary_uses_topic_title():
    summarizer = DailySummarizer(
        topic=TopicConfig(
            slug="security",
            name="Security Radar",
            title_en="Security Daily",
            title_zh="安全日报",
        )
    )

    result = _run_async(
        summarizer.generate_summary(
            [_make_item(1)],
            date="2026-04-25",
            total_fetched=3,
            language="en",
        )
    )

    assert result.startswith("# Security Daily - 2026-04-25")


def test_generate_webhook_item_renders_single_item_detail():
    summarizer = DailySummarizer()

    result = summarizer.generate_webhook_item(
        _make_item(1),
        language="en",
        index=1,
        total=2,
    )

    assert result.startswith("Item 1/2")
    assert "## [Important Item 1](https://example.com/items/1)" in result
    assert "Summary for item 1." in result
    assert "**Tags**: `#AI`, `#News`" in result


def test_generate_webhook_item_includes_discussion_link_when_distinct():
    summarizer = DailySummarizer()
    item = _make_item(1)
    item.metadata["discussion_url"] = "https://news.ycombinator.com/item?id=1"

    result = summarizer.generate_webhook_item(
        item,
        language="en",
        index=1,
        total=1,
    )

    assert "tester · Apr 25, 08:00 · [Discussion](https://news.ycombinator.com/item?id=1)" in result


def test_generate_webhook_item_omits_discussion_link_when_same_as_item_url():
    summarizer = DailySummarizer()
    item = _make_item(1)
    item.metadata["discussion_url"] = item.url

    result = summarizer.generate_webhook_item(
        item,
        language="en",
        index=1,
        total=1,
    )

    assert "[Discussion](https://example.com/items/1)" not in result


def test_generate_webhook_item_uses_localized_discussion_label():
    summarizer = DailySummarizer()
    item = _make_item(1)
    item.metadata["discussion_url"] = "https://www.reddit.com/r/python/comments/abc123/test/"

    result = summarizer.generate_webhook_item(
        item,
        language="zh",
        index=1,
        total=1,
    )

    assert "[社区讨论](https://www.reddit.com/r/python/comments/abc123/test/)" in result


def test_generate_summary_zh_uses_localized_selection_header_and_numeric_date():
    summarizer = DailySummarizer()
    item = _make_item(1)

    result = _run_async(
        summarizer.generate_summary(
            [item],
            date="2026-04-25",
            total_fetched=10,
            language="zh",
        )
    )

    assert "> 从 10 条内容中筛选出 1 条重要资讯。" in result
    assert "rss · tester · 4月25日 08:00" in result
    assert "From 10 items" not in result
    assert "Apr 25, 08:00" not in result


def test_generate_empty_summary_zh_uses_localized_analyzed_line():
    summarizer = DailySummarizer()

    result = _run_async(
        summarizer.generate_summary(
            [],
            date="2026-04-25",
            total_fetched=10,
            language="zh",
        )
    )

    assert "> 已分析 10 条内容，但没有达到重要性阈值的条目。" in result
    assert "Analyzed 10 items" not in result


def test_generate_summary_can_append_bilingual_render_without_ai_calls():
    summarizer = DailySummarizer(
        SummaryConfig(
            include_bilingual=True,
            bilingual_secondary_language="zh",
            include_summary=True,
            max_detailed_items=1,
            compact_remaining=True,
            show_scores=False,
        )
    )
    item = _make_item(1)
    item.metadata["title_en"] = "English Title"
    item.metadata["title_zh"] = "中文标题"
    item.metadata["detailed_summary_en"] = "English summary."
    item.metadata["detailed_summary_zh"] = "中文摘要。"

    result = _run_async(
        summarizer.generate_summary(
            [item],
            date="2026-04-25",
            total_fetched=10,
            language="en",
        )
    )

    assert "本文下方附有中文版。" in result
    assert "# AI CTO Daily - 2026-04-25" in result
    assert "# AI CTO 日报 - 2026-04-25" in result
    assert "[English Title](https://example.com/items/1)" in result
    assert "[中文标题](https://example.com/items/1)" in result
    assert "English summary." in result
    assert "中文摘要。" in result



def test_generate_summary_can_force_bilingual_primary_language():
    summarizer = DailySummarizer(
        SummaryConfig(
            include_bilingual=True,
            bilingual_primary_language="zh",
            bilingual_secondary_language="en",
            show_scores=False,
        ),
        topic=TopicConfig(
            slug="gaming",
            name="Gaming Radar",
            title_en="Gaming Daily",
            title_zh="\u6e38\u620f\u65e5\u62a5",
        ),
    )
    item = _make_item(1)
    item.metadata["title_en"] = "English Title"
    item.metadata["title_zh"] = "\u4e2d\u6587\u6807\u9898"
    item.metadata["detailed_summary_en"] = "English summary."
    item.metadata["detailed_summary_zh"] = "\u4e2d\u6587\u6458\u8981\u3002"

    result = _run_async(
        summarizer.generate_summary(
            [item],
            date="2026-04-25",
            total_fetched=10,
            language="en",
        )
    )

    assert result.find("# \u6e38\u620f\u65e5\u62a5 - 2026-04-25") < result.find("# Gaming Daily - 2026-04-25")
    assert result.find("[\u4e2d\u6587\u6807\u9898](https://example.com/items/1)") < result.find("[English Title](https://example.com/items/1)")


def test_gaming_summary_uses_highlights_label():
    summarizer = DailySummarizer(
        SummaryConfig(
            include_cto_takeaway=True,
            show_scores=False,
        ),
        topic=TopicConfig(
            slug="gaming",
            name="Gaming Radar",
            title_en="Gaming Daily",
            title_zh="\u6e38\u620f\u65e5\u62a5",
        ),
    )
    item = _make_item(1)
    item.metadata["cto_takeaway_zh"] = "\u8fd9\u6761\u65b0\u95fb\u7684\u4eae\u70b9\u662f\u73a9\u5bb6\u53cd\u9988\u548c\u9996\u53d1\u70ed\u5ea6\u3002"

    result = _run_async(
        summarizer.generate_summary(
            [item],
            date="2026-04-25",
            total_fetched=10,
            language="zh",
        )
    )

    assert "## \u4eae\u70b9\u901f\u9012" in result
    assert "CTO" not in result.split("## \u4eae\u70b9\u901f\u9012", 1)[0]

def test_generate_summary_can_skip_opening_summary_and_keep_cto_takeaway():
    summarizer = DailySummarizer(
        SummaryConfig(
            include_summary=False,
            include_cto_takeaway=True,
            include_bilingual=False,
            show_scores=False,
        )
    )
    item = _make_item(1)
    item.metadata["cto_takeaway_en"] = "Prioritize rollout governance before broad adoption."

    result = _run_async(
        summarizer.generate_summary(
            [item],
            date="2026-04-25",
            total_fetched=10,
            language="en",
        )
    )

    assert "## Summary" not in result
    assert "## CTO Takeaway" in result
    assert "Prioritize rollout governance before broad adoption." in result


def test_generate_summary_includes_disclosure():
    summarizer = DailySummarizer(
        SummaryConfig(
            disclosure="由 AI 生成，人类审核。",
            include_summary=False,
            include_cto_takeaway=False,
            include_bilingual=False,
            show_scores=False,
        )
    )

    result = _run_async(
        summarizer.generate_summary(
            [_make_item(1)],
            date="2026-04-25",
            total_fetched=10,
            language="en",
        )
    )

    assert "> 由 AI 生成，人类审核。" in result


def test_generate_summary_uses_localized_ai_cto_takeaway():
    summarizer = DailySummarizer(
        SummaryConfig(
            include_summary=False,
            include_cto_takeaway=True,
            include_bilingual=False,
            show_scores=False,
        )
    )
    item = _make_item(1)
    item.metadata["cto_takeaway_zh"] = "先把治理、成本和试点指标定清楚。"

    result = _run_async(
        summarizer.generate_summary(
            [item],
            date="2026-04-25",
            total_fetched=10,
            language="zh",
        )
    )

    assert "CTO" in result
    assert "先把治理、成本和试点指标定清楚。" in result
