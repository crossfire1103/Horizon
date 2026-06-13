from datetime import datetime, timezone

import httpx

from src.models import SourceType, SteamConfig
from src.scrapers.steam import SteamScraper


async def _fetch_with_transport(transport: httpx.MockTransport):
    async with httpx.AsyncClient(transport=transport) as client:
        scraper = SteamScraper(
            SteamConfig(enabled=True, max_items=5),
            client,
        )
        return await scraper.fetch(
            datetime(2026, 6, 13, tzinfo=timezone.utc)
        )


def test_steam_scraper_fetches_new_release_with_review_summary():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search/results/":
            return httpx.Response(
                200,
                json={
                    "success": 1,
                    "results_html": '<a data-ds-appid="12345" href="/app/12345/Test_Game/">Game</a>',
                    "total_count": 1,
                    "start": 0,
                },
            )
        if request.url.path == "/api/appdetails":
            if request.url.params.get("l") == "schinese":
                return httpx.Response(
                    200,
                    json={
                        "12345": {
                            "success": True,
                            "data": {
                                "type": "game",
                                "steam_appid": 12345,
                                "name": "\u6d4b\u8bd5\u6e38\u620f",
                            },
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "12345": {
                        "success": True,
                        "data": {
                            "type": "game",
                            "steam_appid": 12345,
                            "name": "Test Game",
                            "short_description": "A small but sharp indie game.",
                            "developers": ["Tiny Studio"],
                            "publishers": ["Tiny Studio"],
                            "genres": [{"description": "Indie"}, {"description": "Adventure"}],
                            "release_date": {"coming_soon": False, "date": "13 Jun, 2026"},
                            "is_free": False,
                            "price_overview": {"final_formatted": "$9.99"},
                        },
                    }
                },
            )
        if request.url.path == "/appreviews/12345":
            return httpx.Response(
                200,
                json={
                    "success": 1,
                    "query_summary": {
                        "review_score": 8,
                        "review_score_desc": "Very Positive",
                        "total_positive": 90,
                        "total_negative": 10,
                        "total_reviews": 100,
                    },
                },
            )
        return httpx.Response(404)

    items = __import__("asyncio").run(
        _fetch_with_transport(httpx.MockTransport(handler))
    )

    assert len(items) == 1
    item = items[0]
    assert item.source_type == SourceType.STEAM
    assert item.title == "Steam new release: Test Game"
    assert item.metadata["steam_appid"] == "12345"
    assert item.metadata["steam_name_zh"] == "\u6d4b\u8bd5\u6e38\u620f"
    assert item.metadata["title_zh"] == "Steam \u65b0\u6e38\uff1a\u6d4b\u8bd5\u6e38\u620f"
    assert item.metadata["review_score_desc"] == "Very Positive"
    assert item.metadata["positive_ratio"] == 0.9
    assert "90 positive / 100 total" in item.content
    assert "Steam official Chinese name: \u6d4b\u8bd5\u6e38\u620f" in item.content
