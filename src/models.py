"""Core data models for Horizon."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, HttpUrl, Field, field_validator, model_validator


class SourceType(str, Enum):
    """Supported information source types."""

    GITHUB = "github"
    HACKERNEWS = "hackernews"
    STEAM = "steam"
    RSS = "rss"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    OPENBB = "openbb"
    OSSINSIGHT = "ossinsight"


class ContentItem(BaseModel):
    """Unified content item model from any source."""

    id: str  # Format: {source}:{subtype}:{native_id}
    source_type: SourceType
    title: str
    url: HttpUrl
    content: Optional[str] = None
    author: Optional[str] = None
    published_at: datetime
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # AI analysis results
    ai_score: Optional[float] = None  # 0-10 importance score
    ai_reason: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_tags: List[str] = Field(default_factory=list)


class AIProvider(str, Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    ALI = "ali"
    GEMINI = "gemini"
    DOUBAO = "doubao"
    MINIMAX = "minimax"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


class AIConfig(BaseModel):
    """AI client configuration."""

    provider: AIProvider
    model: str
    base_url: Optional[str] = None
    api_key_env: str
    temperature: float = 0.3
    max_tokens: int = 4096
    throttle_sec: float = 0.0
    analysis_concurrency: int = 1
    enrichment_concurrency: int = 1
    languages: List[str] = Field(default_factory=lambda: ["en"])
    # Azure OpenAI specific; required when provider == AZURE
    azure_endpoint_env: Optional[str] = None
    api_version: Optional[str] = None


class GitHubSourceConfig(BaseModel):
    """GitHub source configuration."""

    type: str  # "user_events", "repo_releases", etc.
    username: Optional[str] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    enabled: bool = True


class HackerNewsConfig(BaseModel):
    """Hacker News configuration."""

    enabled: bool = True
    fetch_top_stories: int = 30
    min_score: int = 100


class RSSSourceConfig(BaseModel):
    """RSS feed source configuration."""

    name: str
    url: HttpUrl
    enabled: bool = True
    category: Optional[str] = None


class RedditSubredditConfig(BaseModel):
    """Configuration for monitoring a specific subreddit."""

    subreddit: str
    enabled: bool = True
    sort: str = "hot"  # hot, new, top, rising
    time_filter: str = (
        "day"  # hour, day, week, month, year, all (only for top/controversial)
    )
    fetch_limit: int = 25
    min_score: int = 10


class RedditUserConfig(BaseModel):
    """Configuration for monitoring a specific Reddit user."""

    username: str  # without u/ prefix
    enabled: bool = True
    sort: str = "new"
    fetch_limit: int = 10


class RedditConfig(BaseModel):
    """Reddit source configuration."""

    enabled: bool = True
    subreddits: List[RedditSubredditConfig] = Field(default_factory=list)
    users: List[RedditUserConfig] = Field(default_factory=list)
    fetch_comments: int = 5  # top comments per post, 0 to disable


class TelegramChannelConfig(BaseModel):
    """Configuration for monitoring a specific Telegram channel."""

    channel: str  # channel username, e.g. "zaihuapd"
    enabled: bool = True
    fetch_limit: int = 20


class TelegramConfig(BaseModel):
    """Telegram source configuration."""

    enabled: bool = True
    channels: List[TelegramChannelConfig] = Field(default_factory=list)


class TwitterConfig(BaseModel):
    """Twitter source configuration via Apify."""

    enabled: bool = True
    apify_token_env: str = "APIFY_TOKEN"
    actor_id: str = "altimis~scweet"
    users: List[str] = Field(default_factory=list)
    fetch_limit: int = 10
    fetch_reply_text: bool = False
    max_replies_per_tweet: int = 3
    max_tweets_to_expand: int = 10
    reply_min_likes: int = 0


class OpenBBWatchlist(BaseModel):
    """A named watchlist of tickers fetched from one OpenBB provider.

    Each watchlist produces one news.company() call per run, so group
    symbols by provider rather than creating one watchlist per symbol.
    """

    name: str
    symbols: List[str] = Field(default_factory=list)
    enabled: bool = True
    provider: str = "yfinance"
    fetch_limit: int = 20
    category: Optional[str] = None


class OpenBBConfig(BaseModel):
    """OpenBB Platform source configuration.

    Uses the installed `openbb` SDK to fetch news and filings for a set of
    tickers. The SDK is an optional dependency; if it is not installed the
    scraper will no-op with a console warning rather than crash the run.

    Provider credentials (FMP, Benzinga, Polygon, Intrinio, Tiingo, etc.)
    are resolved by openbb from environment variables / its own user
    settings file, so Horizon does not need to pass them explicitly.
    """

    enabled: bool = True
    watchlists: List[OpenBBWatchlist] = Field(default_factory=list)
    fetch_filings: bool = False
    filings_provider: str = "sec"


class OSSInsightConfig(BaseModel):
    """OSS Insight trending repos source configuration.

    Pulls top star-gain repositories from the OSS Insight public API and
    emits them as ContentItems. Optional `keywords` filter limits results
    to repos whose description, repo name, or collection names contain at
    least one of the listed substrings (case-insensitive). Leave
    `keywords` empty to ingest everything trending in the configured
    languages.
    """

    enabled: bool = False
    period: str = "past_24_hours"  # past_24_hours, past_28_days
    languages: List[str] = Field(
        default_factory=lambda: ["All", "Python", "TypeScript"]
    )
    keywords: List[str] = Field(default_factory=list)
    min_stars: int = 5
    max_items: int = 30


class SteamConfig(BaseModel):
    """Steam Store source configuration.

    Uses Steam Store search for recently released games, then enriches each app
    with Store app details and Steam review summary data.
    """

    enabled: bool = False
    fetch_new_releases: bool = True
    max_items: int = 30
    candidate_count: int = 200
    country: str = "US"
    language: str = "english"
    category: str = "steam-new-releases"
    new_release_window_days: int = 7
    min_total_reviews: int = 0
    min_positive_ratio: Optional[float] = None
    include_free: bool = True
    include_early_access: bool = True


class SourcesConfig(BaseModel):
    """All sources configuration."""

    github: List[GitHubSourceConfig] = Field(default_factory=list)
    hackernews: HackerNewsConfig = Field(default_factory=HackerNewsConfig)
    rss: List[RSSSourceConfig] = Field(default_factory=list)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    twitter: Optional[TwitterConfig] = None
    steam: SteamConfig = Field(default_factory=SteamConfig)
    openbb: Optional[OpenBBConfig] = None
    ossinsight: OSSInsightConfig = Field(default_factory=OSSInsightConfig)


class WebhookConfig(BaseModel):
    """Webhook notification configuration."""

    url_env: Optional[str] = (
        None  # Environment variable name containing the webhook URL
    )
    request_body: Optional[Union[str, dict, list]] = (
        None  # POST body: real JSON object or string with #{key} placeholders; if empty, will use GET
    )
    headers: Optional[str] = None  # Custom headers, "Key: Value" per line
    delivery: str = "summary"  # summary, or summary_and_items
    overview_position: str = "first"  # For summary_and_items: first, or last
    platform: str = "generic"  # generic, feishu, lark, dingtalk, slack, discord
    layout: str = "markdown"  # markdown, or collapsible
    fallback_layout: str = (
        "markdown"  # Layout to use when the requested layout is unsupported
    )
    languages: Optional[List[str]] = (
        None  # Optional language filter for webhook delivery; defaults to all AI languages
    )
    enabled: bool = False

    @field_validator("delivery")
    @classmethod
    def validate_delivery(cls, v: str) -> str:
        allowed = {"summary", "summary_and_items"}
        if v not in allowed:
            raise ValueError(f"webhook.delivery must be one of {allowed}, got '{v}'")
        return v

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        allowed = {"generic", "feishu", "lark", "dingtalk", "slack", "discord"}
        if v not in allowed:
            raise ValueError(f"webhook.platform must be one of {allowed}, got '{v}'")
        return v

    @field_validator("layout")
    @classmethod
    def validate_layout(cls, v: str) -> str:
        allowed = {"markdown", "collapsible"}
        if v not in allowed:
            raise ValueError(f"webhook.layout must be one of {allowed}, got '{v}'")
        return v

    @field_validator("fallback_layout")
    @classmethod
    def validate_fallback_layout(cls, v: str) -> str:
        allowed = {"markdown", "collapsible"}
        if v not in allowed:
            raise ValueError(
                f"webhook.fallback_layout must be one of {allowed}, got '{v}'"
            )
        return v

    @field_validator("overview_position")
    @classmethod
    def validate_overview_position(cls, v: str) -> str:
        allowed = {"first", "last"}
        if v not in allowed:
            raise ValueError(
                f"webhook.overview_position must be one of {allowed}, got '{v}'"
            )
        return v


class EmailConfig(BaseModel):
    """Email configuration for updates/subscriptions."""

    imap_server: str
    imap_port: int = 993
    imap_enabled: bool = True
    smtp_server: str
    smtp_port: int = 465
    smtp_username: Optional[str] = None
    email_address: str
    password_env: str = "EMAIL_PASSWORD"
    sender_name: str = "AI CTO Daily"
    subscribe_keyword: str = "SUBSCRIBE"
    unsubscribe_keyword: str = "UNSUBSCRIBE"
    enabled: bool = False


class FilteringConfig(BaseModel):
    """Content filtering configuration."""

    ai_score_threshold: float = 7.0
    time_window_hours: int = 24


class SummaryConfig(BaseModel):
    """Markdown report rendering configuration."""

    disclosure: str = "\u7531 AI \u751f\u6210\uff0c\u4eba\u7c7b\u5ba1\u6838\u3002"
    include_summary: bool = False
    include_cto_takeaway: bool = False
    max_detailed_items: int = 0  # 0 means render every selected item in detail
    show_scores: bool = True
    compact_remaining: bool = False
    compact_sentence_limit: int = 2
    include_bilingual: bool = False
    bilingual_primary_language: Optional[str] = None
    bilingual_secondary_language: str = "zh"
    cto_takeaway_ai_enabled: bool = False
    cto_takeaway_ai_items: int = 3


class PromptConfig(BaseModel):
    """Optional prompt overrides for one topic.

    Leave any field empty to use the built-in prompt for that step. User
    prompts are Python ``str.format`` templates and must keep the placeholders
    used by their corresponding built-in prompt.
    """

    analysis_system: Optional[str] = None
    analysis_user: Optional[str] = None
    concept_system: Optional[str] = None
    concept_user: Optional[str] = None
    enrichment_system: Optional[str] = None
    enrichment_user: Optional[str] = None
    takeaway_system: Optional[str] = None
    takeaway_user: Optional[str] = None
    topic_dedup_system: Optional[str] = None
    topic_dedup_user: Optional[str] = None
    translation_system: Optional[str] = None
    translation_user: Optional[str] = None


class TopicConfig(BaseModel):
    """One configurable briefing topic/theme.

    Topics are peer briefing products. Each topic should define its own source,
    filtering, and summary settings while sharing global AI, email, webhook,
    scheduler, and artifact configuration.
    """

    slug: str
    name: str
    enabled: bool = True
    title_en: Optional[str] = None
    title_zh: Optional[str] = None
    description: str = ""
    audience: str = ""
    relevance_prompt: str = ""
    cto_prompt_focus: str = ""
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    sources: Optional[SourcesConfig] = None
    filtering: Optional[FilteringConfig] = None
    summary: Optional[SummaryConfig] = None
    publishing: Optional["PublishingConfig"] = None

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("topic.slug is required")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
        if any(ch not in allowed for ch in normalized):
            raise ValueError("topic.slug may only contain lowercase letters, numbers, hyphen, and underscore")
        return normalized


class ArtifactConfig(BaseModel):
    """Debug artifact and AI replay configuration."""

    enabled: bool = False
    root_dir: str = "data/runs"
    cache_ai_calls: bool = True
    replay_ai_calls: bool = False


class ScheduleConfig(BaseModel):
    """Local long-running schedule configuration."""

    enabled: bool = False
    timezone: str = "Asia/Shanghai"
    daily_times: List[str] = Field(default_factory=lambda: ["09:00"])
    interval_minutes: Optional[int] = None
    hours: Optional[int] = 30
    run_on_start: bool = False
    prevent_overlap: bool = True

    @field_validator("daily_times")
    @classmethod
    def validate_daily_times(cls, values: List[str]) -> List[str]:
        for value in values:
            parts = value.split(":")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"schedule.daily_times entries must be HH:MM, got '{value}'")
            hour, minute = int(parts[0]), int(parts[1])
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                raise ValueError(f"schedule.daily_times entries must be HH:MM, got '{value}'")
        return values

    @field_validator("interval_minutes")
    @classmethod
    def validate_interval_minutes(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("schedule.interval_minutes must be positive")
        return value

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("schedule.hours must be positive")
        return value

    @model_validator(mode="after")
    def validate_schedule_shape(self) -> "ScheduleConfig":
        if self.enabled and self.interval_minutes is None and not self.daily_times:
            raise ValueError("schedule.daily_times is required when interval_minutes is not set")
        return self


class WeChatPublishingConfig(BaseModel):
    """WeChat Official Account publishing configuration."""

    enabled: bool = False
    auto_create_draft_on_run: bool = True
    appid_env: str = "WECHAT_APP_ID"
    secret_env: str = "WECHAT_APP_SECRET"
    author: str = "AI CTO Daily"
    cover_image: str = "assets/wechat-cover.png"
    default_digest: str = "\u4eca\u65e5 AI CTO \u6280\u672f\u60c5\u62a5\u901f\u9012"
    publish_mode: str = "draft"
    review_reminder_enabled: bool = False
    review_reminder_recipients: List[str] = Field(default_factory=list)
    review_reminder_subject: str = "AI CTO Daily is ready for WeChat review"
    review_url: str = "http://127.0.0.1:8765"

    @field_validator("publish_mode")
    @classmethod
    def validate_publish_mode(cls, v: str) -> str:
        allowed = {"draft", "publish"}
        if v not in allowed:
            raise ValueError(f"wechat.publish_mode must be one of {allowed}, got '{v}'")
        return v


class PublishingConfig(BaseModel):
    """Publishing integrations."""

    wechat: WeChatPublishingConfig = Field(default_factory=WeChatPublishingConfig)


class Config(BaseModel):
    """Main configuration model."""

    version: str = "1.0"
    active_topic: str = "ai-cto"
    topics: List[TopicConfig] = Field(default_factory=list)
    ai: AIConfig
    sources: SourcesConfig
    filtering: FilteringConfig
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)
    email: Optional[EmailConfig] = None
    webhook: Optional[WebhookConfig] = None

    @model_validator(mode="after")
    def validate_active_topic(self) -> "Config":
        if self.topics:
            enabled_slugs = {topic.slug for topic in self.topics if topic.enabled}
            all_slugs = {topic.slug for topic in self.topics}
            if self.active_topic not in all_slugs:
                raise ValueError(f"active_topic '{self.active_topic}' was not found in topics")
            if self.active_topic not in enabled_slugs:
                raise ValueError(f"active_topic '{self.active_topic}' is disabled")
        return self

    def get_active_topic(self) -> TopicConfig:
        """Return the active topic, falling back to the legacy AI CTO topic."""
        for topic in self.topics:
            if topic.slug == self.active_topic:
                return topic
        return TopicConfig(
            slug="ai-cto",
            name="AI CTO",
            title_en="AI CTO Daily",
            title_zh="AI CTO \u65e5\u62a5",
            description="AI technology briefing for CTOs and engineering leaders.",
            audience="CTOs, VP Engineering, platform leaders, and senior AI practitioners.",
            relevance_prompt=(
                "This radar is AI-focused. Score 7+ only when the item is directly about AI/ML, "
                "LLMs, AI agents, model releases, AI infrastructure, AI inference/serving, "
                "AI developer tools, AI safety/security, AI product/platform changes, or "
                "technical research that directly advances or evaluates AI systems."
            ),
            cto_prompt_focus=(
                "Focus on enterprise technology strategy, architecture, engineering productivity, "
                "platform reliability, AI infrastructure cost, security/compliance, vendor strategy, "
                "team capability, and roadmap decisions."
            ),
        )

    def scoped_to_active_topic(self) -> "Config":
        """Return a config copy with the active topic's own settings applied.

        Top-level sources/filtering/summary remain as a legacy fallback so
        older single-topic configs continue to run.
        """
        topic = self.get_active_topic()
        scoped = self.model_copy(deep=True)
        if topic.sources is not None:
            scoped.sources = topic.sources.model_copy(deep=True)
        if topic.filtering is not None:
            scoped.filtering = topic.filtering.model_copy(deep=True)
        if topic.summary is not None:
            scoped.summary = topic.summary.model_copy(deep=True)
        if topic.publishing is not None:
            scoped.publishing = self._merge_publishing_override(
                scoped.publishing,
                topic.publishing,
            )
        scoped.active_topic = topic.slug
        return scoped

    @staticmethod
    def _merge_publishing_override(
        base: PublishingConfig,
        override: PublishingConfig,
    ) -> PublishingConfig:
        """Apply only topic-level publishing fields explicitly present in config."""
        merged = base.model_copy(deep=True)
        if "wechat" in override.model_fields_set:
            for field in override.wechat.model_fields_set:
                setattr(merged.wechat, field, getattr(override.wechat, field))
        return merged
