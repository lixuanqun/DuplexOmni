"""Stage 1 of the data pipeline: scenario-seed construction.

The paper builds ~620K scenario seeds from public text chat corpora
(UltraChat, WildChat, BELLE, COIG, no-robots, OASST2).  A seed is a topic
plus a *target set of interaction patterns*; the downstream Writer turns it
into a natural script and the Director overlays the temporal control tokens.

Pattern sampling probabilities follow the paper's reported coverage
(Sec. 4.1): delayed reasoning 94.3%, shared silence 68.2%,
assistant-initiated turns 50%, overlapping speech 49.8%,
interruption-with-reset 41.9%, backchannel 3.1%.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Pattern(str, Enum):
    """Full-duplex interaction patterns the data pipeline can instantiate."""

    DELAYED_REASONING = "delayed_reasoning"      # [THINK] + <...> while still speaking
    SHARED_SILENCE = "shared_silence"            # [PENDNS]
    ASSISTANT_INITIATED = "assistant_initiated"  # assistant opens the exchange
    OVERLAP = "overlap"                          # ˆ user speech over assistant speech
    INTERRUPTION_RESET = "interruption_reset"    # [CUT] ghost-text + [WAIT]
    BACKCHANNEL = "backchannel"                  # "嗯嗯/yeah" while assistant speaks


# Paper Sec. 4.1 coverage statistics.
PAPER_PATTERN_PROBS: dict[Pattern, float] = {
    Pattern.DELAYED_REASONING: 0.943,
    Pattern.SHARED_SILENCE: 0.682,
    Pattern.ASSISTANT_INITIATED: 0.50,
    Pattern.OVERLAP: 0.498,
    Pattern.INTERRUPTION_RESET: 0.419,
    Pattern.BACKCHANNEL: 0.031,
}

_BACKCHANNELS = {
    "zh": ["嗯嗯", "对对", "是的", "好的", "明白"],
    "en": ["yeah", "right", "got it", "makes sense", "mhm"],
}
_OPENERS = {
    "zh": ["你好呀", "在吗", "嘿，问个事儿", "打扰一下"],
    "en": ["hey there", "got a second", "quick question", "hello"],
}


@dataclass
class ScenarioSeed:
    scenario_id: str
    topic: str
    language: str
    patterns: list[Pattern]
    source_corpus: str = "unknown"
    source_snippet: str = ""
    user_persona: str = ""
    difficulty: int = 1  # 1 (casual) .. 3 (requires deep reasoning / tools)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["patterns"] = [p.value for p in self.patterns]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ScenarioSeed:
        d = dict(d)
        d["patterns"] = [Pattern(p) for p in d["patterns"]]
        return cls(**d)


# --------------------------------------------------------------------------- #
# Corpus readers
# --------------------------------------------------------------------------- #


def iter_corpus_texts(corpus: str | Path) -> Iterator[str]:
    """Yield raw text snippets from a corpus file.

    Supports:
    * ``.jsonl`` — records with ``text``/``content``/``question`` fields, or
      OpenAI-style ``messages: [{role, content}]`` lists;
    * ``.json``  — a list of such records;
    * ``.txt``   — one snippet per paragraph (blank-line separated).
    """
    path = Path(corpus)
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = _record_text(rec)
            if text:
                yield text
    elif path.suffix == ".json":
        for rec in json.loads(path.read_text(encoding="utf-8")):
            text = _record_text(rec)
            if text:
                yield text
    else:
        for para in path.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if len(para) >= 12:
                yield para


def _record_text(rec: dict) -> str:
    if isinstance(rec.get("messages"), list):
        parts = [
            str(m.get("content", ""))
            for m in rec["messages"]
            if m.get("role") in {"user", "assistant"}
        ]
        return "\n".join(p for p in parts if p)
    for key in ("text", "content", "question", "instruction"):
        if isinstance(rec.get(key), str) and rec[key].strip():
            return rec[key]
    return ""


def detect_language(text: str) -> str:
    """Rough CJK ratio test: zh if >15% CJK code points, else en."""
    if not text:
        return "en"
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk / max(len(text), 1) > 0.15 else "en"


# --------------------------------------------------------------------------- #
# Seed sampling
# --------------------------------------------------------------------------- #


def sample_patterns(rng: random.Random, *, probs: dict[Pattern, float] | None = None) -> list[Pattern]:
    """Independently sample each pattern with its coverage probability.

    Guarantees at least one pattern so that every seed teaches some
    full-duplex behaviour (paper: 90.7% of samples cover >= 2 patterns).
    """
    probs = probs or PAPER_PATTERN_PROBS
    chosen = [p for p, prob in probs.items() if rng.random() < prob]
    if not chosen:
        chosen = [rng.choice(list(Pattern))]
    return chosen


def topic_from_text(text: str, *, max_words: int = 12) -> str:
    """Collapse a corpus snippet into a short topic phrase."""
    first_line = text.splitlines()[0].strip() if text else ""
    lang = detect_language(first_line)
    if lang == "zh":
        topic = first_line[:16]
    else:
        words = first_line.split()
        topic = " ".join(words[:max_words])
    # rstrip over a trailing-punctuation character set is intended here
    return topic.rstrip("。！？？!?.,；;：: \t") or "something interesting"  # noqa: B005


def sample_scenario_seeds(
    corpus: str | Path,
    n: int,
    *,
    seed: int | None = None,
    source_corpus: str | None = None,
    probs: dict[Pattern, float] | None = None,
) -> list[ScenarioSeed]:
    """Build ``n`` scenario seeds from a corpus file."""
    rng = random.Random(seed)
    texts = list(iter_corpus_texts(corpus))
    if not texts:
        raise ValueError(f"no usable text found in corpus {corpus!r}")
    src = source_corpus or Path(corpus).stem

    seeds: list[ScenarioSeed] = []
    for i in range(n):
        snippet = rng.choice(texts)
        language = detect_language(snippet)
        patterns = sample_patterns(rng, probs=probs)
        # deep reasoning difficulty when delayed reasoning is requested
        difficulty = 3 if Pattern.DELAYED_REASONING in patterns and rng.random() < 0.5 else rng.choice([1, 2])
        seeds.append(
            ScenarioSeed(
                scenario_id=f"seed-{src}-{seed if seed is not None else 0}-{i:06d}",
                topic=topic_from_text(snippet),
                language=language,
                patterns=patterns,
                source_corpus=src,
                source_snippet=snippet[:500],
                user_persona=_persona(rng, language),
                difficulty=difficulty,
                metadata={},
            )
        )
    return seeds


def _persona(rng: random.Random, language: str) -> str:
    personas = {
        "zh": ["学生", "上班族", "游客", "研究者", "家长"],
        "en": ["student", "office worker", "tourist", "researcher", "parent"],
    }
    return rng.choice(personas.get(language, personas["en"]))


def pattern_coverage_stats(samples: Iterable[ScenarioSeed]) -> dict[str, float]:
    """Empirical pattern coverage, mirroring the paper's Table stats."""
    samples = list(samples)
    n = max(len(samples), 1)
    counts = {p.value: 0 for p in Pattern}
    multi = 0
    for s in samples:
        if len(s.patterns) >= 2:
            multi += 1
        for p in s.patterns:
            counts[p.value] += 1
    stats = {k: v / n for k, v in counts.items()}
    stats["multi_pattern>=2"] = multi / n
    return stats
