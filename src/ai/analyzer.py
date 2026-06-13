"""Content analysis using AI."""

import asyncio
import json
import re
from typing import List, Optional
from tenacity import RetryError
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

from .client import AIClient
from .prompts import CONTENT_ANALYSIS_USER, build_content_analysis_system, get_prompt
from .utils import detect_original_language, parse_json_response
from ..models import ContentItem, TopicConfig
from ..storage.artifacts import ai_trace_context

DEFAULT_THROTTLE_SEC = 0.0


class ContentAnalyzer:
    """Analyzes content items using AI to determine importance."""

    def __init__(self, ai_client: AIClient, topic: TopicConfig | None = None):
        self.client = ai_client
        self.topic = topic

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """Try multiple strategies to extract a JSON object from an AI response.

        Returns the parsed dict, or None if all strategies fail.
        """
        return parse_json_response(response)

    def _get_throttle_sec(self) -> float:
        """Return the configured inter-item throttle, clamped to zero or above."""
        config = getattr(self.client, "config", None)
        throttle_sec = getattr(config, "throttle_sec", DEFAULT_THROTTLE_SEC)
        return max(throttle_sec, 0.0)

    def _get_concurrency(self) -> int:
        """Return the configured analysis concurrency, clamped to 1 or above."""
        config = getattr(self.client, "config", None)
        concurrency = getattr(config, "analysis_concurrency", 1)
        return max(concurrency, 1)

    async def analyze_batch(self, items: List[ContentItem]) -> List[ContentItem]:
        throttle_sec = self._get_throttle_sec()
        concurrency = self._get_concurrency()
        semaphore = asyncio.Semaphore(concurrency)

        async def _process(item: ContentItem, index: int, progress_task) -> ContentItem:
            async with semaphore:
                try:
                    await self._analyze_item(item)
                except Exception as e:
                    detail = self._format_analysis_error(e)
                    print(f"Error analyzing item {item.id}: {detail}")
                    item.ai_score = 0.0
                    item.ai_reason = "Analysis failed"
                    item.ai_summary = item.title
                if throttle_sec > 0 and index < len(items) - 1:
                    await asyncio.sleep(throttle_sec)
            progress.advance(progress_task)
            return item

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing", total=len(items))
            coros = [
                _process(item, i, task) for i, item in enumerate(items)
            ]
            analyzed_items = await asyncio.gather(*coros)

        return analyzed_items

    @staticmethod
    def _format_analysis_error(error: Exception) -> str:
        if isinstance(error, RetryError):
            last = error.last_attempt.exception()
            if last is not None:
                return f"{type(last).__name__}: {last}"
        return f"{type(error).__name__}: {error}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10)
    )
    async def _analyze_item(self, item: ContentItem) -> None:
        """Analyze a single content item.

        Args:
            item: Content item to analyze (modified in-place)
        """
        original_sample = f"{item.title}\n\n{item.content or ''}"
        item.metadata.setdefault(
            "original_language",
            detect_original_language(original_sample[:4000]),
        )

        # Prepare content section
        content_section = ""
        if item.content:
            # Split off comments if present
            content_text = item.content
            if "--- Top Comments ---" in content_text:
                main, comments_part = content_text.split("--- Top Comments ---", 1)
                content_section = f"Content: {main.strip()[:800]}"
            else:
                content_section = f"Content: {content_text[:1000]}"

        # Prepare discussion section (comments, engagement)
        discussion_parts = []
        if item.content and "--- Top Comments ---" in item.content:
            comments_part = item.content.split("--- Top Comments ---", 1)[1]
            discussion_parts.append(f"Community Comments:\n{comments_part[:1500]}")

        meta = item.metadata
        engagement_items = []
        if meta.get("score"):
            engagement_items.append(f"score: {meta['score']}")
        if meta.get("descendants"):
            engagement_items.append(f"{meta['descendants']} comments")
        if meta.get("favorite_count"):
            engagement_items.append(f"{meta['favorite_count']} likes")
        if meta.get("retweet_count"):
            engagement_items.append(f"{meta['retweet_count']} retweets")
        if meta.get("reply_count"):
            engagement_items.append(f"{meta['reply_count']} replies")
        if meta.get("views"):
            engagement_items.append(f"{meta['views']} views")
        if meta.get("bookmarks"):
            engagement_items.append(f"{meta['bookmarks']} bookmarks")
        if meta.get("upvote_ratio"):
            engagement_items.append(f"upvote ratio: {meta['upvote_ratio']:.0%}")
        if engagement_items:
            discussion_parts.append(f"Engagement: {', '.join(engagement_items)}")
        if meta.get("discussion_url"):
            discussion_parts.append(f"Discussion: {meta['discussion_url']}")
        if meta.get("community_note"):
            discussion_parts.append(f"Community Note: {meta['community_note']}")

        discussion_section = "\n".join(discussion_parts) if discussion_parts else ""

        source_parts = [item.source_type.value]
        if meta.get("feed_name"):
            source_parts.append(f"feed={meta['feed_name']}")
        if meta.get("category"):
            source_parts.append(f"category={meta['category']}")
        if meta.get("subreddit"):
            source_parts.append(f"subreddit=r/{meta['subreddit']}")
        source_label = " | ".join(source_parts)

        # Generate user prompt
        user_template = get_prompt(self.topic, "analysis_user", CONTENT_ANALYSIS_USER)
        user_prompt = user_template.format(
            title=item.title,
            source=source_label,
            author=item.author or "Unknown",
            url=str(item.url),
            published_at=item.published_at.isoformat(),
            original_language=item.metadata.get("original_language", "unknown"),
            content_section=content_section,
            discussion_section=discussion_section
        )

        # Get AI completion
        with ai_trace_context(stage="analyze", item_id=item.id, item_title=item.title):
            response = await self.client.complete(
                system=build_content_analysis_system(self.topic),
                user=user_prompt,
            )

        # Parse JSON response with robust fallback
        result = self._parse_json_response(response)
        if result is None:
            print(f"Warning: could not parse analysis response for {item.id}, using defaults")
            item.ai_score = 0.0
            item.ai_reason = "Analysis response parse failed"
            item.ai_summary = item.title
            item.ai_tags = []
            return

        # Update item with analysis results
        item.ai_score = float(result.get("score", 0))
        item.ai_reason = result.get("reason", "")
        item.ai_summary = result.get("summary", item.title)
        item.ai_tags = result.get("tags", [])
