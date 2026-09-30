"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def corpus_file(tmp_path_factory) -> Path:
    """A tiny mixed-language chat corpus."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    records = [
        {
            "messages": [
                {"role": "user", "content": "今天天气怎么样，适合出门爬山吗"},
                {"role": "assistant", "content": "今天多云转晴，适合户外活动"},
            ]
        },
        {"text": "please help me plan a trip to Chengdu with three friends next weekend"},
        {"question": "如何做一道简单的番茄炒蛋？"},
        {"messages": [
            {"role": "user", "content": "my wifi keeps dropping every few minutes"},
            {"role": "assistant", "content": "let's check the router channel settings"},
        ]},
        {"text": "介绍一下大模型全双工语音交互的技术难点"},
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path
