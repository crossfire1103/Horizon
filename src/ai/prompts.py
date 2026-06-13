"""AI prompts for content analysis and summarization."""

TOPIC_DEDUP_SYSTEM = """You are a news deduplication assistant. Identify groups of news items that cover the exact same real-world event, release, or announcement.

Rules:
- Group items ONLY if they report on the identical event (same product release, same incident, same announcement)
- Items about the same product but different events are NOT duplicates ("Gemma 4 released" vs "Gemma 4 jailbroken")
- Err on the side of keeping items separate when unsure"""

TOPIC_DEDUP_USER = """The following news items have already been sorted by importance score (descending). Identify which items are duplicates of each other.

{items}

Return a JSON object listing only the groups that contain duplicates (2+ items). Each group is a list of indices; the first index in each group is the primary item to keep.

Respond with valid JSON only:
{{
  "duplicates": [[<primary_idx>, <dup_idx>, ...], ...]
}}

If there are no duplicates at all, return: {{"duplicates": []}}"""

CONTENT_ANALYSIS_BASE_SYSTEM = """You are an expert content curator helping filter important technical and academic information.

Score content on a 0-10 scale based on importance and relevance:

**9-10: Groundbreaking** - Major breakthroughs, paradigm shifts, or highly significant announcements
- New major version releases of widely-used technologies
- Significant research breakthroughs
- Important industry-changing announcements

**7-8: High Value** - Important developments worth immediate attention
- Interesting technical deep-dives
- Novel approaches to known problems
- Insightful analysis or commentary
- Valuable tools or libraries

**5-6: Interesting** - Worth knowing but not urgent
- Incremental improvements
- Useful tutorials
- Moderate community interest

**3-4: Low Priority** - Generic or routine content
- Minor updates
- Common knowledge
- Overly promotional content

**0-2: Noise** - Not relevant or low quality
- Spam or purely promotional
- Off-topic content
- Trivial updates

Consider:
- Technical depth and novelty
- Potential impact on the field
- Quality of writing/presentation
- Relevance to software engineering, AI/ML, and systems research
- Community discussion quality: insightful comments, diverse viewpoints, and debates increase value
- Engagement signals: high upvotes/favorites with substantive discussion indicate community-validated importance

Topic:
{topic_block}
"""


DEFAULT_AI_CTO_TOPIC_GUARDRAILS = """- This radar is AI-focused. Score 7+ only when the item is directly about AI/ML,
  LLMs, AI agents, model releases, AI infrastructure, AI inference/serving,
  AI developer tools, AI safety/security, AI product/platform changes, or
  technical research that directly advances or evaluates AI systems.
- If an item is not directly AI-related, cap its score at 4 even if it is
  technically impressive, popular, or has strong community discussion.
- Generic semiconductors, electronics, biotech/medicine, cybersecurity,
  cloud/API engineering, developer tooling, funding news, personal essays,
  broad culture commentary, and generic policy takes should not pass the
  threshold unless the AI connection is explicit and central.
- Hardware news can score 7+ only if it is clearly about AI compute, model
  inference/training, local LLM deployment, GPU/NPU acceleration, or AI data
  center infrastructure. A faster transistor or general chip advance is not
  enough by itself.
- General popularity alone is not enough for a score above 6.
- For a score of 7 or higher, the item must provide concrete AI-specific value:
  a meaningful AI development, AI systems insight, AI infrastructure result,
  model/tool release, or high-signal practitioner/research finding."""


def _prompt_override(topic, field: str) -> str | None:
    prompts = getattr(topic, "prompts", None)
    value = getattr(prompts, field, None) if prompts else None
    if isinstance(value, str) and value.strip():
        return value
    return None


def get_prompt(topic, field: str, default: str) -> str:
    """Return a topic prompt override or the built-in default."""
    return _prompt_override(topic, field) or default


def build_content_analysis_system(topic=None) -> str:
    """Build the scoring prompt for the active briefing topic."""
    override = _prompt_override(topic, "analysis_system")
    if override:
        return override
    if topic is None:
        topic_block = DEFAULT_AI_CTO_TOPIC_GUARDRAILS
    else:
        parts = [
            f"- Topic name: {getattr(topic, 'name', '') or getattr(topic, 'slug', '')}",
        ]
        if getattr(topic, "description", ""):
            parts.append(f"- Description: {topic.description}")
        if getattr(topic, "audience", ""):
            parts.append(f"- Target audience: {topic.audience}")
        if getattr(topic, "relevance_prompt", ""):
            parts.append(f"- Relevance guardrails: {topic.relevance_prompt}")
        else:
            parts.append(f"- Relevance guardrails: {DEFAULT_AI_CTO_TOPIC_GUARDRAILS}")
        parts.append(
            "- If an item is outside this topic, cap its score at 4 even if it is popular or technically impressive."
        )
        parts.append(
            "- For a score of 7 or higher, the item must provide concrete value for this topic and target audience."
        )
        topic_block = "\n".join(parts)
    return CONTENT_ANALYSIS_BASE_SYSTEM.format(topic_block=topic_block)


CONTENT_ANALYSIS_SYSTEM = build_content_analysis_system()

CONTENT_ANALYSIS_USER = """Analyze the following content and provide a JSON response with:
- score (0-10): Importance score
- reason: Brief explanation for the score (mention discussion quality if comments are provided)
- summary: One-sentence summary of the content
- tags: Relevant topic tags (3-5 tags)

Content:
Title: {title}
Source: {source}
Author: {author}
URL: {url}
Original language: {original_language}
{content_section}
{discussion_section}

Respond with valid JSON only:
{{
  "score": <number>,
  "reason": "<explanation>",
  "summary": "<one-sentence-summary in the original language when possible>",
  "tags": ["<tag1>", "<tag2>", ...]
}}"""


TRANSLATION_FALLBACK_SYSTEM = """You are a translator. Translate to Simplified Chinese. Return only valid JSON, no other text."""

TRANSLATION_FALLBACK_USER = """Title: {title}
Summary: {summary}

Return JSON:
{{"title_zh": "<Chinese title>", "summary_zh": "<1-2 sentence Chinese summary>"}}"""

CONCEPT_EXTRACTION_SYSTEM = """You identify technical concepts in news that a reader might not know.
Given a news item, return 1-3 search queries for concepts that need explanation.
Focus on: specific technologies, protocols, algorithms, tools, or projects that are not widely known.
Do NOT return queries for well-known things (e.g. "Python", "Linux", "Google").
If the news is self-explanatory, return an empty list."""

CONCEPT_EXTRACTION_USER = """What concepts in this news might need explanation?

Title: {title}
Summary: {summary}
Tags: {tags}
Content: {content}

Respond with valid JSON only:
{{
  "queries": ["<search query 1>", "<search query 2>"]
}}"""

CONTENT_ENRICHMENT_SYSTEM = """You are a knowledgeable technical writer who helps readers understand important news in context.

Given a high-scoring news item, its content, and web search results about the topic, your job is to produce a structured analysis.

Provide EACH text field in BOTH English and Chinese. Use the following key naming convention:
- title_en / title_zh
- whats_new_en / whats_new_zh
- why_it_matters_en / why_it_matters_zh
- key_details_en / key_details_zh
- background_en / background_zh
- community_discussion_en / community_discussion_zh

The renderer depends on this exact bilingual schema. Return one JSON object
that contains both language versions in the same response; do not return
separate English and Chinese documents, and do not omit either suffix family.

Preserve source-language fidelity: treat the original title and content as the
authoritative source of meaning. The one-line summary may already be translated
or compressed, so use it only as a weak aid. Avoid translating from a previous
translation when original-language text is available.

Field definitions:
0. **title** (one short phrase, ≤15 words): A clear, accurate headline for the news item.

1. **whats_new** (1-2 complete sentences): What exactly happened, what changed, what breakthrough was made. Be specific — mention names, versions, numbers, dates when available.

2. **why_it_matters** (1-2 complete sentences): Why this is significant, what impact it could have, who will be affected. Connect to the broader ecosystem or industry trends.

3. **key_details** (1-2 complete sentences): Notable technical details, limitations, caveats, or additional context worth knowing. Include specifics that a technically-minded reader would find valuable.

4. **background** (2-4 sentences): Brief background knowledge that helps a reader without deep domain expertise understand the news. Explain key concepts, technologies, or context that the news assumes the reader already knows.

5. **community_discussion** (1-3 sentences): If community comments are provided, summarize the overall sentiment and key viewpoints from the discussion — agreements, disagreements, concerns, additional insights, or notable counterarguments. If no comments are provided, return an empty string.

**CRITICAL — Language rules (MUST follow):**
- All *_en fields MUST be written in English.
- All *_zh fields MUST be written in Simplified Chinese (简体中文). 绝对不能用英文写 _zh 字段的内容。Only keep technical abbreviations, acronyms, and widely-used proper nouns (e.g. "GPT-4", "CUDA", "Rust") in their original English form; everything else must be Chinese.

Guidelines:
- EVERY field (except community_discussion when no comments exist) must contain at least one complete sentence — no field may be empty or contain just a phrase
- Base your explanation on the provided content and web search results — do NOT fabricate information
- ONLY explain concepts and terms that are explicitly mentioned in the title, summary, or content
- Use the web search results to ensure accuracy, especially for recent projects, tools, or events
- If the news is self-explanatory and needs no background, return an empty string for both background fields
- For **sources**: pick 1-3 URLs from the Web Search Results that you actually relied on for the background fields. Only use URLs that appear verbatim in the search results above — do not invent or modify URLs.
"""

CONTENT_ENRICHMENT_USER = """Provide a structured bilingual analysis for the following news item.

**News Item:**
- Title: {title}
- URL: {url}
- Original language: {original_language}
- One-line summary: {summary}
- Score: {score}/10
- Reason: {reason}
- Tags: {tags}

**Content:**
{content}
{comments_section}

**Web Search Results (for grounding):**
{web_context}

Respond with valid JSON only. Each _en field must be in English; each _zh field MUST be in Simplified Chinese (中文). Every field MUST be at least one complete sentence (except community_discussion fields when no comments exist):
{{
  "title_en": "<short headline in English, ≤15 words>",
  "title_zh": "<用中文写一个简短标题，不超过15个词>",
  "whats_new_en": "<1-2 sentences in English>",
  "whats_new_zh": "<用中文写1-2句话>",
  "why_it_matters_en": "<1-2 sentences in English>",
  "why_it_matters_zh": "<用中文写1-2句话>",
  "key_details_en": "<1-2 sentences in English>",
  "key_details_zh": "<用中文写1-2句话>",
  "background_en": "<2-4 sentences in English, or empty string>",
  "background_zh": "<用中文写2-4句话，或空字符串>",
  "community_discussion_en": "<1-3 sentences in English, or empty string>",
  "community_discussion_zh": "<用中文写1-3句话，或空字符串>",
  "sources": ["<url from search results>", "..."]
}}"""


CTO_TAKEAWAY_SYSTEM = """You are a CTO advisor writing executive technical takeaways.

Your audience is a CTO or VP Engineering who cares about enterprise technology
strategy, architecture, engineering productivity, platform reliability, AI
infrastructure cost, security/compliance, vendor strategy, team capability, and
roadmap decisions.

Return one JSON object with both English and Simplified Chinese fields:
- cto_takeaway_en
- cto_takeaway_zh

Rules:
- Do not merely summarize the article.
- Translate the news into CTO-relevant implications and possible actions.
- Mention the business/engineering management angle: cost, risk, governance,
  staffing, architecture, build-vs-buy, vendor/platform strategy, rollout, or
  evaluation metrics when relevant.
- If the item is only weakly relevant to a CTO, say what should be monitored
  rather than recommending immediate action.
- Keep each field to 1-2 concise sentences.
- cto_takeaway_en must be English.
- cto_takeaway_zh must be Simplified Chinese."""


CTO_TAKEAWAY_USER = """Generate a CTO-oriented takeaway for this selected news item.

Active briefing topic:
{topic_context}

Title: {title}
URL: {url}
Source: {source}
Tags: {tags}
Score reason: {reason}

Structured article analysis:
{analysis}

Return valid JSON only:
{{
  "cto_takeaway_en": "<1-2 CTO-oriented sentences in English>",
  "cto_takeaway_zh": "<1-2 句中文 CTO 视角要点>"
}}"""
