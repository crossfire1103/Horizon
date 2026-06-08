"""Render Markdown reports into WeChat-friendly HTML."""

from __future__ import annotations

import re

import markdown
from bs4 import BeautifulSoup


CONTAINER_STYLE = (
    "font-size:16px;line-height:1.75;color:#1f2937;"
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
)
H1_STYLE = "font-size:24px;line-height:1.35;font-weight:700;color:#111827;margin:0 0 18px;"
H2_STYLE = "font-size:20px;line-height:1.4;font-weight:700;color:#111827;margin:28px 0 12px;border-left:4px solid #2563eb;padding-left:10px;"
H3_STYLE = "font-size:17px;line-height:1.45;font-weight:700;color:#111827;margin:22px 0 10px;"
P_STYLE = "margin:10px 0;color:#1f2937;"
QUOTE_STYLE = "margin:14px 0;padding:10px 12px;background:#f3f6fb;border-left:4px solid #93a4bd;color:#475467;"
CODE_STYLE = "font-family:Menlo,Consolas,monospace;background:#f2f4f7;border-radius:4px;padding:1px 4px;font-size:14px;"
PRE_STYLE = "font-family:Menlo,Consolas,monospace;background:#111827;color:#e5e7eb;border-radius:6px;padding:12px;white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.55;"
HR_STYLE = "border:none;border-top:1px solid #e5e7eb;margin:24px 0;"
URL_TEXT_STYLE = "margin:10px 0;color:#1f2937;word-break:break-all;"


def markdown_to_wechat_html(markdown_text: str) -> str:
    """Convert Markdown to inline-styled HTML suitable for WeChat drafts."""
    html = markdown.markdown(
        markdown_text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )
    soup = BeautifulSoup(html, "html.parser")
    _strip_unsafe_elements(soup)
    _remove_details_sections(soup)
    _convert_links_to_plain_urls(soup)
    _remove_empty_nodes(soup)
    _flatten_lists(soup)
    _inline_styles(soup)
    body = "".join(str(child) for child in soup.contents)
    return f'<section style="{CONTAINER_STYLE}">{body}</section>'


def extract_title(markdown_text: str, default: str = "AI CTO Daily") -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return normalize_title(_clean_inline(stripped[2:]))[:64] or default
    return default


def normalize_title(title: str) -> str:
    """Normalize legacy report branding in published titles."""
    stripped = title.strip()
    if stripped.startswith("Horizon Daily"):
        return "AI CTO Daily" + stripped.removeprefix("Horizon Daily")
    if stripped.startswith("Horizon 每日速递"):
        return "AI CTO 日报" + stripped.removeprefix("Horizon 每日速递")
    return stripped


def extract_digest(markdown_text: str, default: str, limit: int = 120) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("---"):
            continue
        stripped = stripped.lstrip("> ").strip()
        text = _clean_text(stripped)
        if text:
            return text[:limit]
    return default[:limit]


def append_full_version_note(markdown_text: str, full_version_url: str | None) -> str:
    """Append a WeChat-friendly note pointing to the full link-rich version."""
    if not full_version_url:
        return markdown_text
    note = (
        "\n\n---\n\n"
        "完整链接版请点击文末「阅读原文」。\n\n"
    )
    return markdown_text.rstrip() + note


def _strip_unsafe_elements(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "iframe", "object", "embed"]):
        tag.decompose()


def _remove_details_sections(soup: BeautifulSoup) -> None:
    """Drop reference/detail blocks in the WeChat version.

    The normal web/Markdown report keeps the full reference list. In WeChat,
    large clusters of external URLs are noisy and can trigger content checks,
    so the draft points readers to the full version instead.
    """
    for details in soup.find_all("details"):
        details.decompose()


def _convert_links_to_plain_urls(soup: BeautifulSoup) -> None:
    for link in list(soup.find_all("a")):
        href = str(link.get("href") or "").strip()
        label = link.get_text(" ", strip=True)
        parent = link.parent
        if not href or href.startswith("#"):
            link.replace_with(label)
            continue

        link.replace_with(label)
        if not _is_allowed_url(href) or not _should_emit_plain_url(label):
            continue

        url_node = soup.new_tag("p")
        url_node["class"] = "wechat-plain-url"
        url_node["style"] = URL_TEXT_STYLE
        url_node.string = f"{_plain_url_prefix(label)}{href}"

        if parent and parent.name in {"p", "li"}:
            parent.insert_after(url_node)


def _inline_styles(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(True):
        if tag.name == "h1":
            tag["style"] = H1_STYLE
        elif tag.name == "h2":
            tag["style"] = H2_STYLE
        elif tag.name == "h3":
            tag["style"] = H3_STYLE
        elif tag.name == "p":
            if "wechat-plain-url" in (tag.get("class") or []):
                tag["style"] = URL_TEXT_STYLE
            else:
                tag["style"] = tag.get("style", "") + P_STYLE
        elif tag.name == "blockquote":
            tag["style"] = QUOTE_STYLE
        elif tag.name == "code":
            tag["style"] = CODE_STYLE
        elif tag.name == "pre":
            tag["style"] = PRE_STYLE
            for code in tag.find_all("code"):
                code["style"] = ""
        elif tag.name == "a":
            # WeChat draft content is strict about links in article body.
            # Keep this as a defensive fallback; normal links are converted
            # to text plus explicit URL paragraphs before styling.
            tag.unwrap()
        elif tag.name == "hr":
            tag["style"] = HR_STYLE
        elif tag.name == "table":
            tag["style"] = "border-collapse:collapse;width:100%;margin:12px 0;"
        elif tag.name in {"th", "td"}:
            tag["style"] = "border:1px solid #e5e7eb;padding:6px 8px;"


def _is_allowed_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "#"))


def _should_emit_plain_url(label: str) -> bool:
    normalized = re.sub(r"\s+", " ", label).strip().lower()
    return normalized in {"original", "原文", "discussion", "讨论", "社区讨论"}


def _plain_url_prefix(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label).strip().lower()
    if normalized in {"original", "discussion"}:
        return "Link: "
    return "链接："


def _remove_empty_nodes(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(["p", "li"])):
        if not tag.get_text(" ", strip=True) and not tag.find(["img", "br"]):
            tag.decompose()


def _flatten_lists(soup: BeautifulSoup) -> None:
    """Convert HTML lists to plain paragraphs for WeChat compatibility.

    WeChat's editor can interpret blank lines inside list HTML as empty list
    items. Plain paragraphs with textual bullets/numbers are less elegant in
    raw HTML, but they paste and render far more predictably.
    """
    for list_tag in list(soup.find_all(["ul", "ol"])):
        replacements = []
        ordered = list_tag.name == "ol"
        index = 1
        for li in list_tag.find_all("li", recursive=False):
            text = li.get_text(" ", strip=True)
            if not text:
                continue
            prefix = f"{index}. " if ordered else "• "
            index += 1
            p = soup.new_tag("p")
            p["class"] = "wechat-list-line"
            p.string = prefix + text
            replacements.append(p)

        for replacement in replacements:
            list_tag.insert_before(replacement)
        list_tag.decompose()


def _clean_text(value: str) -> str:
    value = _clean_inline(value)
    value = re.sub(r"[>#]+", " ", value)
    value = re.sub(r"\s+-\s+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _clean_inline(value: str) -> str:
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[`*_]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()
