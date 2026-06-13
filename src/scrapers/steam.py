"""Steam Store scraper for new release review signals."""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any, List

import httpx

from .base import BaseScraper
from ..models import ContentItem, SourceType, SteamConfig


STEAM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}


class SteamScraper(BaseScraper):
    """Fetch recently released Steam games with review summary data."""

    SEARCH_URL = "https://store.steampowered.com/search/results/"
    APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
    APPREVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

    def __init__(self, config: SteamConfig, http_client: httpx.AsyncClient):
        super().__init__({"config": config}, http_client)
        self.steam_config = config

    async def fetch(self, since: datetime) -> List[ContentItem]:
        if not self.steam_config.enabled or not self.steam_config.fetch_new_releases:
            return []

        appids = await self._fetch_candidate_appids()
        window_since = datetime.now(timezone.utc) - timedelta(
            days=max(self.steam_config.new_release_window_days, 1)
        )
        items: list[ContentItem] = []
        for appid in appids:
            try:
                item = await self._fetch_app_item(appid, window_since)
            except httpx.HTTPError:
                continue
            if item:
                items.append(item)
            if len(items) >= self.steam_config.max_items:
                break
        return items

    async def _fetch_candidate_appids(self) -> list[str]:
        target = max(
            self.steam_config.candidate_count,
            self.steam_config.max_items * 4,
            25,
        )
        page_size = 100
        seen: set[str] = set()
        appids: list[str] = []

        for start in range(0, target, page_size):
            response = await self.client.get(
                self.SEARCH_URL,
                params={
                    "query": "",
                    "start": start,
                    "count": min(page_size, target - start),
                    "sort_by": "Released_DESC",
                    "category1": "998",  # Games
                    "json": 1,
                    "infinite": 1,
                    "cc": self.steam_config.country,
                    "l": self.steam_config.language,
                },
                headers=STEAM_HEADERS,
                follow_redirects=True,
            )
            response.raise_for_status()
            results_html = str(response.json().get("results_html") or "")
            before = len(appids)
            for match in re.finditer(r'data-ds-appid="(\d+)"', results_html):
                appid = match.group(1)
                if appid not in seen:
                    seen.add(appid)
                    appids.append(appid)
            if len(appids) == before or len(appids) >= target:
                break

        return appids[:target]

    async def _fetch_app_item(self, appid: str, since: datetime) -> ContentItem | None:
        details = await self._fetch_appdetails(appid)
        if not details or details.get("type") != "game":
            return None
        zh_details = await self._fetch_appdetails(appid, language="schinese")

        release_date = self._parse_release_date(details.get("release_date") or {})
        if not release_date or release_date < since:
            return None

        if not self.steam_config.include_free and details.get("is_free"):
            return None
        if not self.steam_config.include_early_access and self._is_early_access(details):
            return None

        reviews = await self._fetch_review_summary(appid)
        total_reviews = int(reviews.get("total_reviews") or 0)
        positive_reviews = int(reviews.get("total_positive") or 0)
        positive_ratio = positive_reviews / total_reviews if total_reviews > 0 else None
        if total_reviews < self.steam_config.min_total_reviews:
            return None
        if (
            self.steam_config.min_positive_ratio is not None
            and positive_ratio is not None
            and positive_ratio < self.steam_config.min_positive_ratio
        ):
            return None

        title = str(details.get("name") or f"Steam app {appid}")
        title_zh = self._localized_name(title, zh_details)
        url = f"https://store.steampowered.com/app/{appid}/"
        content = self._build_content(details, reviews, positive_ratio, title_zh)
        metadata = {
            "steam_appid": appid,
            "steam_name": title,
            "steam_name_zh": title_zh,
            "category": self.steam_config.category,
            "steam_signal": "new_release",
            "developers": details.get("developers") or [],
            "publishers": details.get("publishers") or [],
            "genres": [genre.get("description") for genre in details.get("genres") or []],
            "release_date": release_date.date().isoformat(),
            "is_free": bool(details.get("is_free")),
            "price": self._price_text(details),
            "review_score": reviews.get("review_score"),
            "review_score_desc": reviews.get("review_score_desc"),
            "total_reviews": total_reviews,
            "total_positive": positive_reviews,
            "total_negative": int(reviews.get("total_negative") or 0),
            "positive_ratio": positive_ratio,
            "tags": [genre.get("description") for genre in details.get("genres") or []],
        }
        if title_zh:
            metadata["title_zh"] = f"Steam \u65b0\u6e38\uff1a{title_zh}"

        return ContentItem(
            id=self._generate_id("steam", "new_release", appid),
            source_type=SourceType.STEAM,
            title=f"Steam new release: {title}",
            url=url,
            content=content,
            author=", ".join(details.get("developers") or []) or "Steam",
            published_at=release_date,
            metadata=metadata,
        )

    async def _fetch_appdetails(
        self,
        appid: str,
        *,
        language: str | None = None,
    ) -> dict[str, Any] | None:
        response = await self.client.get(
            self.APPDETAILS_URL,
            params={
                "appids": appid,
                "cc": self.steam_config.country,
                "l": language or self.steam_config.language,
            },
            headers=STEAM_HEADERS,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        app_payload = payload.get(str(appid)) or {}
        if not app_payload.get("success"):
            return None
        data = app_payload.get("data")
        return data if isinstance(data, dict) else None

    async def _fetch_review_summary(self, appid: str) -> dict[str, Any]:
        response = await self.client.get(
            self.APPREVIEWS_URL.format(appid=appid),
            params={
                "json": 1,
                "filter": "summary",
                "language": "all",
                "purchase_type": "all",
                "num_per_page": 0,
            },
            headers=STEAM_HEADERS,
            follow_redirects=True,
        )
        response.raise_for_status()
        summary = response.json().get("query_summary")
        return summary if isinstance(summary, dict) else {}

    @staticmethod
    def _parse_release_date(raw: dict[str, Any]) -> datetime | None:
        date_text = str(raw.get("date") or "").strip()
        if not date_text or raw.get("coming_soon"):
            return None
        for fmt in ("%d %b, %Y", "%b %d, %Y", "%d %B, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(date_text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _price_text(details: dict[str, Any]) -> str:
        if details.get("is_free"):
            return "Free"
        overview = details.get("price_overview") or {}
        return str(overview.get("final_formatted") or overview.get("initial_formatted") or "")

    @staticmethod
    def _is_early_access(details: dict[str, Any]) -> bool:
        genres = details.get("genres") or []
        categories = details.get("categories") or []
        haystack = " ".join(
            str(item.get("description") or "") for item in [*genres, *categories]
        ).lower()
        return "early access" in haystack

    @staticmethod
    def _localized_name(
        fallback_name: str,
        localized_details: dict[str, Any] | None,
    ) -> str | None:
        if not localized_details:
            return None
        name = str(localized_details.get("name") or "").strip()
        if not name or name == fallback_name:
            return None
        return name

    def _build_content(
        self,
        details: dict[str, Any],
        reviews: dict[str, Any],
        positive_ratio: float | None,
        title_zh: str | None = None,
    ) -> str:
        ratio_text = f"{positive_ratio:.0%}" if positive_ratio is not None else "n/a"
        developers = ", ".join(details.get("developers") or []) or "Unknown"
        publishers = ", ".join(details.get("publishers") or []) or "Unknown"
        genres = ", ".join(
            genre.get("description") for genre in details.get("genres") or []
            if genre.get("description")
        ) or "Unknown"
        description = html.unescape(
            re.sub("<[^<]+?>", " ", str(details.get("short_description") or ""))
        ).strip()
        return "\n".join(
            [
                f"Steam app id: {details.get('steam_appid')}",
                f"Steam official Chinese name: {title_zh or ''}",
                f"Release date: {(details.get('release_date') or {}).get('date', '')}",
                f"Developer: {developers}",
                f"Publisher: {publishers}",
                f"Genres: {genres}",
                f"Price: {self._price_text(details)}",
                f"Reviews: {reviews.get('review_score_desc') or 'No review summary'}; "
                f"{reviews.get('total_positive') or 0} positive / "
                f"{reviews.get('total_reviews') or 0} total ({ratio_text} positive)",
                f"Description: {description}",
            ]
        )
