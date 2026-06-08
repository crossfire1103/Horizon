"""Shared AI utility functions."""

import json
import re
from typing import Optional


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def detect_original_language(text: str) -> str:
    """Best-effort language label for preserving source-language context.

    This intentionally stays lightweight: it only distinguishes CJK-heavy text
    from mostly English/Latin text, and returns ``unknown`` for empty content.
    The label is used as prompt context, not as a hard routing decision.
    """
    stripped = text.strip()
    if not stripped:
        return "unknown"

    cjk = len(_CJK_RE.findall(stripped))
    ascii_letters = len(_ASCII_LETTER_RE.findall(stripped))
    if cjk >= 8 and cjk >= ascii_letters * 0.25:
        return "zh"
    if ascii_letters > 0:
        return "en"
    return "unknown"


def parse_json_response(response: str) -> Optional[dict]:
    """Try multiple strategies to extract a JSON object from an AI response.

    Returns the parsed dict, or None if all strategies fail.
    """
    text = response.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract from ```json ... ``` code block
    if "```json" in text:
        try:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    # Strategy 3: extract from ``` ... ``` code block
    if "```" in text:
        try:
            json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    # Strategy 4: find the first { ... } block using brace matching
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break

    # Strategy 5: regex extraction as last resort
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None
