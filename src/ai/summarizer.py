"""Daily summary generation — pure programmatic rendering."""

import re
from typing import List, Optional

from ..models import ContentItem, SummaryConfig


_CJK = r"[\u4e00-\u9fff\u3400-\u4dbf]"
_ASCII = r"[A-Za-z0-9]"


def _pangu(text: str) -> str:
    """Insert a space between CJK and ASCII letters/digits (Pangu spacing)."""
    text = re.sub(rf"({_CJK})({_ASCII})", r"\1 \2", text)
    text = re.sub(rf"({_ASCII})({_CJK})", r"\1 \2", text)
    return text


LABELS = {
    "en": {
        "header": "Horizon Daily",
        "source": "Source",
        "background": "Background",
        "discussion": "Discussion",
        "references": "References",
        "tags": "Tags",
        "summary": "Summary",
        "cto_takeaway": "CTO Takeaway",
        "detailed_items": "Detailed Items",
        "brief_mentions": "Brief Mentions",
        "original": "Original",
        "bilingual_note": "本文下方附有中文版。",
        "selected_items": "From {total} items, {selected} important content pieces were selected",
        "empty_analyzed": "Analyzed {total} items, but none met the importance threshold.",
        "empty_body": (
            "No significant developments today. This might indicate:\n"
            "- A quiet day in your tracked sources\n"
            "- The AI score threshold is too high\n"
            "- Your information sources need expansion\n\n"
            "Consider:\n"
            "1. Lowering the `ai_score_threshold` in config.json\n"
            "2. Adding more diverse information sources\n"
            "3. Checking if the AI model is working correctly\n"
        ),
    },
    "zh": {
        "header": "Horizon 每日速递",
        "source": "来源",
        "background": "背景",
        "discussion": "社区讨论",
        "references": "参考链接",
        "tags": "标签",
        "selected_items": "从 {total} 条内容中筛选出 {selected} 条重要资讯。",
        "empty_analyzed": "已分析 {total} 条内容，但没有达到重要性阈值的条目。",
        "empty_body": (
            "今日暂无重要动态，可能原因：\n"
            "- 今天关注的信息源较平静\n"
            "- AI 评分阈值设置过高\n"
            "- 信息源种类有待扩充\n\n"
            "建议：\n"
            "1. 在 config.json 中降低 `ai_score_threshold`\n"
            "2. 添加更多多样化的信息源\n"
            "3. 检查 AI 模型是否正常工作\n"
        ),
    },
}

LABELS["zh"].update(
    {
        "header": "Horizon 每日速递",
        "source": "来源",
        "background": "背景",
        "discussion": "社区讨论",
        "references": "参考链接",
        "tags": "标签",
        "summary": "摘要",
        "cto_takeaway": "CTO 要点",
        "detailed_items": "详细资讯",
        "brief_mentions": "简要资讯",
        "original": "原文",
        "bilingual_note": "本文下方附有英文版本。",
        "selected_items": "从 {total} 条内容中筛选出 {selected} 条重要资讯。",
        "empty_analyzed": "已分析 {total} 条内容，但没有达到重要性阈值的条目。",
    }
)


class DailySummarizer:
    """Generates daily Markdown summaries from pre-analyzed content items."""

    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig()

    async def generate_summary(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
    ) -> str:
        """Generate daily summary in Markdown format.

        Items are rendered in score-descending order (already sorted by orchestrator).

        Args:
            items: High-scoring content items (already enriched)
            date: Date string (YYYY-MM-DD)
            total_fetched: Total number of items fetched before filtering
            language: Output language, either "en" or "zh"

        Returns:
            str: Markdown formatted summary
        """
        if self.config.include_bilingual:
            secondary = self.config.bilingual_secondary_language
            if secondary and secondary != language:
                primary = self._generate_summary_single(
                    items,
                    date,
                    total_fetched,
                    language=language,
                    bilingual_note=True,
                )
                secondary_summary = self._generate_summary_single(
                    items,
                    date,
                    total_fetched,
                    language=secondary,
                    bilingual_note=False,
                )
                return primary.rstrip() + "\n\n---\n\n" + secondary_summary

        return self._generate_summary_single(
            items,
            date,
            total_fetched,
            language=language,
            bilingual_note=False,
        )

    def _generate_summary_single(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
        bilingual_note: bool = False,
    ) -> str:
        labels = LABELS.get(language, LABELS["en"])

        if not items:
            return self._generate_empty_summary(date, total_fetched, labels)

        header = (
            f"# {labels['header']} - {date}\n\n"
            f"> {labels['selected_items'].format(total=total_fetched, selected=len(items))}\n\n"
        )
        if bilingual_note:
            header += f"> {labels['bilingual_note']}\n\n"

        front_sections = []
        if self.config.include_summary:
            front_sections.append(self._generate_opening_summary(items, labels, language))
        if self.config.include_cto_takeaway:
            front_sections.append(self._generate_cto_takeaway(items, labels, language))

        # TOC
        toc_entries = []
        for i, item in enumerate(items):
            _t = item.metadata.get(f"title_{language}") or item.title
            t = str(_t).replace("[", "(").replace("]", ")")
            if language == "zh":
                t = _pangu(t)
            score = f" \u2b50\ufe0f {item.ai_score or '?'}/10" if self.config.show_scores else ""
            toc_entries.append(f"{i + 1}. [{t}](#item-{i + 1}){score}")
        toc = "\n".join(toc_entries) + "\n\n---\n\n"

        detailed_count = self._detailed_count(items)
        use_sections = self.config.compact_remaining and detailed_count < len(items)
        parts = []
        for i, item in enumerate(items):
            if use_sections and i == 0:
                parts.append(f"## {labels['detailed_items']}\n\n")
            if use_sections and i == detailed_count:
                parts.append(f"## {labels['brief_mentions']}\n\n")
            compact = self.config.compact_remaining and i >= detailed_count
            parts.append(self._format_item(item, labels, language, i + 1, compact=compact))

        return header + "".join(front_sections) + "---\n\n" + toc + "".join(parts)

    def generate_webhook_overview(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
    ) -> str:
        """Generate a compact overview for multi-message webhook delivery."""
        labels = LABELS.get(language, LABELS["en"])
        if not items:
            return self._generate_empty_summary(date, total_fetched, labels)

        if language == "zh":
            header = (
                f"# {labels['header']} - {date}\n\n"
                f"> 从 {total_fetched} 条内容中筛选出 {len(items)} 条重要资讯。\n\n"
                "下面会按新闻逐条发送详情，你可以只看感兴趣的标题。\n\n"
            )
        else:
            header = (
                f"# {labels['header']} - {date}\n\n"
                f"> Selected {len(items)} important items from {total_fetched} fetched items.\n\n"
                "Details will be sent item by item so you can read only the topics you care about.\n\n"
            )

        entries = []
        for i, item in enumerate(items, start=1):
            title = str(item.metadata.get(f"title_{language}") or item.title).replace("[", "(").replace("]", ")")
            if language == "zh":
                title = _pangu(title)
            score = item.ai_score or "?"
            entries.append(f"{i}. [{title}]({item.url}) \u2b50\ufe0f {score}/10")

        return header + "\n".join(entries)

    def generate_webhook_item(
        self,
        item: ContentItem,
        language: str,
        index: int,
        total: int,
    ) -> str:
        """Generate one item message for multi-message webhook delivery."""
        labels = LABELS.get(language, LABELS["en"])
        prefix = f"第 {index}/{total} 条\n\n" if language == "zh" else f"Item {index}/{total}\n\n"
        return prefix + self._format_item(item, labels, language, index).rstrip("-\n ")

    def _format_item(
        self,
        item: ContentItem,
        labels: dict,
        language: str,
        index: int,
        compact: bool = False,
    ) -> str:
        """Format a single ContentItem into Markdown."""
        _title = item.metadata.get(f"title_{language}") or item.title
        title = str(_title).replace("[", "(").replace("]", ")")
        url = str(item.url)
        meta = item.metadata

        summary = (
            meta.get(f"detailed_summary_{language}")
            or meta.get("detailed_summary")
            or item.ai_summary
            or ""
        )
        background = meta.get(f"background_{language}") or meta.get("background") or ""
        discussion = (
            meta.get(f"community_discussion_{language}")
            or meta.get("community_discussion")
            or ""
        )

        if language == "zh":
            title = _pangu(title)
            summary = _pangu(summary)
            background = _pangu(background)
            discussion = _pangu(discussion)

        # Source line with parts joined by " · ", link appended at end
        source_type = item.source_type.value
        source_parts = [source_type]
        if meta.get("subreddit"):
            source_parts.append(f"r/{meta['subreddit']}")
        if meta.get("feed_name"):
            source_parts.append(meta["feed_name"])
        else:
            source_parts.append(item.author or "unknown")
        if item.published_at:
            if language == "zh":
                source_parts.append(
                    f"{item.published_at.month}月{item.published_at.day}日 "
                    f"{item.published_at:%H:%M}"
                )
            else:
                day = item.published_at.strftime("%d").lstrip("0")
                source_parts.append(item.published_at.strftime(f"%b {day}, %H:%M"))
        source_line = " \u00b7 ".join(source_parts)  # ·

        discussion_url = meta.get("discussion_url")
        if discussion_url:
            discussion_url = str(discussion_url)
            if discussion_url != url:
                source_line += f' · [{labels["discussion"]}]({discussion_url})'
        source_line += f' · [{labels["original"]}]({url})'

        score = f" \u2b50\ufe0f {item.ai_score or '?'}/10" if self.config.show_scores else ""
        heading = "###" if compact else "##"
        lines = [f'<a id="item-{index}"></a>', f"{heading} [{title}]({url}){score}", ""]

        if compact:
            summary = self._limit_sentences(summary, self.config.compact_sentence_limit)
            lines += [summary, "", source_line, "", "---"]
            return "\n".join(lines) + "\n\n"

        lines += [summary, "", source_line]

        if background:
            lines.append("")
            lines.append(f"**{labels['background']}**: {background}")

        sources = meta.get("sources") or []
        if sources:
            items_html = "".join(f'<li><a href="{s["url"]}">{s["title"]}</a></li>\n' for s in sources)
            lines += [
                "",
                f'<details><summary>{labels["references"]}</summary>\n<ul>\n{items_html}\n</ul>\n</details>',
            ]

        if discussion:
            lines.append("")
            lines.append(f"**{labels['discussion']}**: {discussion}")

        if item.ai_tags:
            tags_str = ", ".join([f"`#{t}`" for t in item.ai_tags])
            lines.append("")
            lines.append(f"**{labels['tags']}**: {tags_str}")

        lines.append("")
        lines.append("---")

        return "\n".join(lines) + "\n\n"

    def _detailed_count(self, items: List[ContentItem]) -> int:
        max_items = self.config.max_detailed_items
        if max_items <= 0:
            return len(items)
        return min(max_items, len(items))

    def _generate_opening_summary(self, items: List[ContentItem], labels: dict, language: str) -> str:
        lines = [f"## {labels['summary']}", ""]
        for item in items[: min(5, len(items))]:
            title = self._display_title(item, language)
            summary = self._display_summary(item, language)
            if language == "zh":
                title = _pangu(title)
                summary = _pangu(summary)
            lines.append(f"- [{title}]({item.url}): {summary}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _generate_cto_takeaway(self, items: List[ContentItem], labels: dict, language: str) -> str:
        lines = [f"## {labels['cto_takeaway']}", ""]
        limit = self.config.cto_takeaway_ai_items if self.config.cto_takeaway_ai_items > 0 else 3
        for item in items[: min(limit, len(items))]:
            title = self._display_title(item, language)
            takeaway = self._display_cto_takeaway(item, language)
            if language == "zh":
                title = _pangu(title)
                takeaway = _pangu(takeaway)
                lines.append(f"- [{title}]({item.url})：{takeaway}")
            else:
                lines.append(f"- [{title}]({item.url}): {takeaway}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _display_cto_takeaway(self, item: ContentItem, language: str) -> str:
        generated = item.metadata.get(f"cto_takeaway_{language}")
        if generated:
            return str(generated)
        summary = self._limit_sentences(self._display_summary(item, language), 1)
        return f"{self._cto_angle(item, language)} {summary}".strip()

    def _cto_angle(self, item: ContentItem, language: str) -> str:
        text = " ".join(
            [
                item.title,
                " ".join(item.ai_tags or []),
                str(item.metadata.get("category", "")),
                self._display_summary(item, language),
            ]
        ).lower()

        def has_term(term: str) -> bool:
            term = term.lower()
            if re.fullmatch(r"[a-z0-9_.+-]+", term):
                return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
            return term in text

        def has_any(terms) -> bool:
            return any(has_term(term) for term in terms)

        ai_terms = (
            "ai",
            "ml",
            "llm",
            "model",
            "agent",
            "inference",
            "vllm",
            "llama.cpp",
            "gemma",
            "qwen",
            "gpu",
            "npu",
            "模型",
            "智能体",
            "推理",
            "大模型",
            "生成",
        )
        infra_terms = (
            "serving",
            "kv cache",
            "speculative",
            "latency",
            "throughput",
            "rtx",
            "oculink",
            "ai factory",
            "data center",
            "部署",
            "算力",
            "延迟",
            "吞吐",
            "加速",
            "数据中心",
        )
        has_ai_context = has_any(ai_terms)
        has_infra_context = has_any(infra_terms)

        if has_any(("security", "privacy", "pii", "de-ident", "prompt injection", "auth", "漏洞", "隐私", "安全", "授权")):
            return (
                "CTO should treat this as a risk, compliance, and platform-control signal;"
                " review data handling, abuse monitoring, and security ownership."
                if language != "zh"
                else "从 CTO 视角看，这是风险、合规与平台治理信号；应检查数据处理、滥用监控和安全责任边界。"
            )
        if has_ai_context and has_infra_context:
            return (
                "CTO should map this to infrastructure cost, latency, capacity planning, and build-vs-buy choices;"
                " ask whether it changes the AI serving roadmap."
                if language != "zh"
                else "从 CTO 视角看，应把它映射到基础设施成本、延迟、容量规划和自建/采购决策；判断是否影响 AI 服务路线图。"
            )
        if has_any(("agent", "claude code", "developer", "tool", "workflow", "graphql", "api", "devtool", "工程", "开发", "工作流", "工具")):
            return (
                "CTO should evaluate whether this can improve engineering throughput, platform reliability, or developer experience;"
                " pilot it behind measurable productivity and quality metrics."
                if language != "zh"
                else "从 CTO 视角看，应评估它是否能提升工程吞吐、平台可靠性或开发者体验；用可衡量的效率和质量指标小范围试点。"
            )
        if has_ai_context and has_any(("benchmark", "release", "open source", "image generation", "multimodal", "基准", "发布", "开源", "多模态")):
            return (
                "CTO should track this as a model portfolio signal;"
                " update evaluation sets, vendor strategy, and internal adoption criteria before changing production defaults."
                if language != "zh"
                else "从 CTO 视角看，这是模型组合信号；在调整生产默认模型前，应更新评测集、供应商策略和内部采用标准。"
            )
        return (
            "CTO should assess whether this changes technical strategy, operating risk, team capability, or near-term roadmap priorities."
            if language != "zh"
            else "从 CTO 视角看，应判断它是否改变技术战略、运营风险、团队能力或近期路线图优先级。"
        )

    @staticmethod
    def _display_title(item: ContentItem, language: str) -> str:
        return str(item.metadata.get(f"title_{language}") or item.title).replace("[", "(").replace("]", ")")

    @staticmethod
    def _display_summary(item: ContentItem, language: str) -> str:
        return str(
            item.metadata.get(f"detailed_summary_{language}")
            or item.metadata.get("detailed_summary")
            or item.ai_summary
            or ""
        )

    @staticmethod
    def _limit_sentences(text: str, limit: int) -> str:
        if limit <= 0:
            return text
        parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
        sentences = [part.strip() for part in parts if part.strip()]
        if len(sentences) <= limit:
            return text
        return " ".join(sentences[:limit])

    def _generate_empty_summary(self, date: str, total_fetched: int, labels: dict) -> str:
        """Generate summary when no high-scoring items were found."""
        return (
            f"# {labels['header']} - {date}\n\n"
            f"> {labels['empty_analyzed'].format(total=total_fetched)}\n\n"
            + labels["empty_body"]
        )
