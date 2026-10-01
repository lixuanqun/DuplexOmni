"""Decide whether an utterance stays in the interaction layer or is delegated.

The interaction layer answers at once. Anything that needs tools, memory, or
multi-step work becomes a task. The lists are intentionally small and
explicit so the default path is predictable without a classifier model.
"""

from __future__ import annotations

import re

_DELEGATE_PHRASES = (
    "计算",
    "算一下",
    "帮我算",
    "等于多少",
    "几点",
    "现在时间",
    "什么时间",
    "今天几号",
    "日期",
    "记住",
    "帮我记",
    "回忆",
    "我让你记",
    "你记了",
)

_EXPR = re.compile(
    r"(?:(?:\d+(?:\.\d+)?)|(?:[()+\-*/%])){3,}"
)


def looks_like_math(text: str) -> bool:
    compact = (
        text.replace("×", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
    )
    if not _EXPR.search(compact):
        return False
    return any(op in compact for op in "+-*/%") and any(ch.isdigit() for ch in compact)


def route_utterance(text: str) -> str:
    """Return ``chat`` or ``delegate``."""
    if looks_like_math(text):
        return "delegate"
    if any(phrase in text for phrase in _DELEGATE_PHRASES):
        return "delegate"
    return "chat"
