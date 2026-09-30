"""Director role of the Writer-Director pipeline (paper Sec. 4.1, App. A).

The Director overlays *temporal control tokens* on a natural script:

* ``[THINK]`` + ``<...>`` fragments   -> delayed reasoning
* ``[PENDNS]``                       -> shared silence
* assistant opener                   -> assistant-initiated turn
* ``ˆ``                              -> overlapping speech onset
* ``[CUT]`` + ghost text + ``[WAIT]``-> interruption with reasoning reset
* ``ˆ`` + tiny acknowledgement       -> backchannel

Two implementations are provided:

* :class:`RuleDirector` — deterministic, offline, implements every pattern
  explicitly (used by tests / demo / as fallback);
* :class:`LLMDirector` — prompts an LLM backend (the paper's approach with
  Qwen3.5-397B-A27B); its output goes through the same grammar and the
  consistency checker, falling back to :class:`RuleDirector` on failure.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..tokens import Event, OverlapOnset, Speaker, TurnStart, parse_script
from .llm import LLMBackend
from .scenario import Pattern, ScenarioSeed
from .synthesis import estimate_duration, is_backchannel
from .writer import Writer, parse_script_lines

__all__ = [
    "UtteranceKind",
    "AnnotatedUtterance",
    "AnnotatedDialog",
    "Director",
    "RuleDirector",
    "LLMDirector",
]

DIRECTOR_SYSTEM = """You are the DIRECTOR in a Writer-Director data pipeline for a
full-duplex spoken dialogue model. Rewrite the script into a temporally
annotated script by inserting control tokens:
- "[THINK]" in an assistant utterance that needs deep reasoning/tool use; the
  assistant KEEPS talking naturally after it (delayed reasoning).
- "<...>" lines AFTER the [THINK] utterance that stream the reasoning result
  fragments progressively (2-4 short fragments, plain text inside the angle
  brackets).
- "ˆ" at the start of a user utterance that begins while the assistant is
  still speaking (overlap).
- "[CUT]" inside an assistant utterance that gets interrupted; everything
  after [CUT] on that line is ghost text kept in history but never spoken;
  end that line with "[WAIT]" to reset the pending reasoning.
- "[PENDnS]" alone on a line for n seconds of shared silence.
Keep every original utterance recognisable. Output ONLY annotated lines
starting with [U] or [A] (or a control-only line)."""


class UtteranceKind(str):
    SPEECH = "speech"
    EVENT = "event"      # control-only utterance (fragments, silence)
    SILENCE = "silence"  # [PENDnS]


@dataclass
class AnnotatedUtterance:
    speaker: Speaker
    text: str
    kind: str = UtteranceKind.SPEECH
    duration_s: float = 0.0
    overlap: bool = False        # starts while the other party still speaks
    backchannel: bool = False
    patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker.value,
            "text": self.text,
            "kind": self.kind,
            "duration_s": round(self.duration_s, 3),
            "overlap": self.overlap,
            "backchannel": self.backchannel,
            "patterns": self.patterns,
        }


@dataclass
class AnnotatedDialog:
    seed: ScenarioSeed
    utterances: list[AnnotatedUtterance]

    @property
    def patterns(self) -> list[str]:
        return [p.value for p in self.seed.patterns]

    def to_script(self) -> str:
        lines = []
        for u in self.utterances:
            tag = "[U]" if u.speaker is Speaker.USER else "[A]"
            body = u.text.strip()
            if not body:
                continue
            if u.kind in (UtteranceKind.EVENT, UtteranceKind.SILENCE):
                lines.append(body)
            else:
                lines.append(f"{tag} {body}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.seed.scenario_id,
            "language": self.seed.language,
            "patterns": self.patterns,
            "utterances": [u.to_dict() for u in self.utterances],
        }


class Director(Protocol):
    def annotate(self, script: str, seed: ScenarioSeed) -> AnnotatedDialog: ...


DurationEstimator = Callable[[str, str], float]


def _default_estimator(text: str, language: str) -> float:
    return estimate_duration(text, language=language)


def _split_at(text: str, fraction: float) -> tuple[str, str]:
    """Split text at the sentence/clause boundary closest to *fraction*."""
    seps = "。！？；.!?;，,"
    target = fraction * len(text)
    best = -1
    best_d = 1e9
    for i, ch in enumerate(text):
        if ch in seps:
            d = abs(i - target)
            if d < best_d:
                best_d = d
                best = i
    if best == -1:
        idx = max(1, int(fraction * len(text)))
        return text[:idx].rstrip(), text[idx:].lstrip()
    return text[: best + 1].strip(), text[best + 1 :].strip()


class RuleDirector:
    """Deterministic rule-based Director.

    Anchor choice is seeded by ``seed.scenario_id`` so annotations are
    reproducible for a given script + scenario.
    """

    def __init__(self, estimator: DurationEstimator | None = None) -> None:
        self.estimator = estimator or _default_estimator

    def annotate(self, script: str, seed: ScenarioSeed) -> AnnotatedDialog:
        rng = random.Random(hash(seed.scenario_id) & 0xFFFF)
        turns = _script_turns(script)
        if len(turns) < 2:
            raise ValueError("script needs at least 2 utterances")

        patterns = set(seed.patterns)
        lang = seed.language
        out: list[AnnotatedUtterance] = []

        if Pattern.ASSISTANT_INITIATED in patterns:
            opener = "嘿，我在呢，想聊点什么？" if lang == "zh" else "hey, I'm here — what's on your mind?"
            out.append(
                AnnotatedUtterance(
                    Speaker.ASSISTANT, opener, duration_s=self.estimator(opener, lang),
                    patterns=[Pattern.ASSISTANT_INITIATED.value],
                )
            )

        n_user = sum(1 for sp, _ in turns if sp is Speaker.USER)
        # anchors ----------------------------------------------------------------
        think_at = self._pick_assistant_anchor(turns, seed, rng)  # index in turns
        interrupt_user_idx = self._pick_interrupt_anchor(turns, rng, avoid_assistant=think_at)
        silence_after = self._pick_silence_anchor(turns, rng)
        overlap_user_idx = self._pick_overlap_anchor(turns, rng, exclude={interrupt_user_idx})
        backchannel_at = self._pick_backchannel_anchor(turns, rng) if n_user >= 2 else -1

        fragments = self._fragments(seed, rng)
        frag_pending = len(fragments)
        used_fragments = 0
        think_aborted = False

        for i, (sp, text) in enumerate(turns):
            pats: list[str] = []
            if sp is Speaker.USER:
                is_interrupt = i == interrupt_user_idx
                is_backch = i == backchannel_at
                # a backchannel always overlaps the assistant it reacts to
                is_overlap = i == overlap_user_idx or is_interrupt or is_backch
                body = text
                if is_backch:
                    body = "嗯嗯" if lang == "zh" else "yeah"
                    pats.append(Pattern.BACKCHANNEL.value)
                if is_overlap:
                    body = "ˆ " + body
                    pats.append(Pattern.OVERLAP.value)
                if is_interrupt:
                    pats.append(Pattern.INTERRUPTION_RESET.value)
                dur = 0.45 if is_backch else self.estimator(body, lang)
                out.append(
                    AnnotatedUtterance(
                        Speaker.USER, body, duration_s=dur, overlap=is_overlap,
                        backchannel=is_backch, patterns=pats,
                    )
                )
                if i == silence_after and Pattern.SHARED_SILENCE in patterns:
                    n = rng.choice([1, 2, 3])
                    out.append(
                        AnnotatedUtterance(
                            Speaker.ASSISTANT, f"[PEND{n}S]", kind=UtteranceKind.SILENCE,
                            duration_s=float(n), patterns=[Pattern.SHARED_SILENCE.value],
                        )
                    )
            else:
                body = text
                if i + 1 == interrupt_user_idx and Pattern.INTERRUPTION_RESET in patterns:
                    spoken, ghost = _split_at(body, 0.55)
                    body = f"{spoken} [CUT] {ghost} [WAIT]" if ghost else f"{spoken} [CUT] [WAIT]"
                    pats.append(Pattern.INTERRUPTION_RESET.value)
                    if think_at != -1 and think_at <= i:
                        # the interruption invalidates a pending reasoning
                        # request: its un-streamed fragments must be dropped
                        frag_pending = 0
                        think_aborted = True
                if i == think_at and Pattern.DELAYED_REASONING in patterns:
                    # insert [THINK] into the spoken part (never into ghost text)
                    spoken_part = body.split(" [CUT] ")[0]
                    rest = body[len(spoken_part):]
                    head, tail = _split_at(spoken_part, 0.35)
                    spoken_part = f"{head} [THINK] {tail}" if tail else f"{head} [THINK]"
                    body = spoken_part + rest
                    pats.append(Pattern.DELAYED_REASONING.value)
                out.append(
                    AnnotatedUtterance(
                        Speaker.ASSISTANT, body,
                        duration_s=self._speech_duration(body, lang),
                        patterns=pats,
                    )
                )
                # stream remaining fragments progressively after each
                # post-[THINK] assistant turn (delayed reasoning)
                if (
                    frag_pending > 0
                    and Pattern.DELAYED_REASONING in patterns
                    and think_at != -1
                    and i > think_at
                    and rng.random() < 0.7
                ):
                    out.append(
                        AnnotatedUtterance(
                            Speaker.ASSISTANT, f"<{fragments[used_fragments]}>",
                            kind=UtteranceKind.EVENT, duration_s=0.0,
                            patterns=[Pattern.DELAYED_REASONING.value],
                        )
                    )
                    used_fragments += 1
                    frag_pending -= 1

        # any un-streamed fragments land at the end (still valid: result
        # arrives late) — unless no reasoning was requested or the request
        # was aborted by an interruption
        if Pattern.DELAYED_REASONING in patterns and not think_aborted:
            for j in range(used_fragments, len(fragments)):
                out.append(
                    AnnotatedUtterance(
                        Speaker.ASSISTANT, f"<{fragments[j]}>", kind=UtteranceKind.EVENT,
                        duration_s=0.0, patterns=[Pattern.DELAYED_REASONING.value],
                    )
                )

        return AnnotatedDialog(seed=seed, utterances=out)

    # -- anchor pickers -------------------------------------------------------- #

    @staticmethod
    def _pick_assistant_anchor(turns: Sequence[tuple[Speaker, str]], seed: ScenarioSeed,
                               rng: random.Random) -> int:
        candidates = [
            i for i, (sp, t) in enumerate(turns)
            if sp is Speaker.ASSISTANT and len(t) >= 24
        ]
        if not candidates:
            return -1
        if seed.difficulty >= 3:
            return candidates[len(candidates) // 2]  # mid-dialogue deep reasoning
        return candidates[0]

    @staticmethod
    def _pick_interrupt_anchor(
        turns: Sequence[tuple[Speaker, str]],
        rng: random.Random,
        *,
        avoid_assistant: int = -1,
    ) -> int:
        """Pick a user turn that interrupts; its assistant predecessor must not
        be the [THINK] anchor, otherwise the abort would orphan the fragments."""
        user_idx = [
            i for i, (sp, _) in enumerate(turns)
            if sp is Speaker.USER and 0 < i < len(turns) - 1 and (i - 1) != avoid_assistant
        ]
        return rng.choice(user_idx) if user_idx else -1

    @staticmethod
    def _pick_overlap_anchor(turns: Sequence[tuple[Speaker, str]], rng: random.Random,
                             *, exclude: set[int]) -> int:
        user_idx = [i for i, (sp, _) in enumerate(turns)
                    if sp is Speaker.USER and i not in exclude and i > 0]
        return rng.choice(user_idx) if user_idx else -1

    @staticmethod
    def _pick_silence_anchor(turns: Sequence[tuple[Speaker, str]], rng: random.Random) -> int:
        user_idx = [i for i, (sp, _) in enumerate(turns) if sp is Speaker.USER]
        return rng.choice(user_idx[:-1]) if len(user_idx) > 1 else -1

    @staticmethod
    def _pick_backchannel_anchor(turns: Sequence[tuple[Speaker, str]], rng: random.Random) -> int:
        # a user turn that directly follows a long assistant turn
        candidates = [
            i for i, (sp, _) in enumerate(turns)
            if sp is Speaker.USER and i > 0 and turns[i - 1][0] is Speaker.ASSISTANT
            and len(turns[i - 1][1]) >= 30
        ]
        return rng.choice(candidates) if candidates else -1

    def _speech_duration(self, text: str, language: str) -> float:
        """Duration of the *spoken* part only (ghost text after [CUT] excluded)."""
        from ..tokens import ControlTokenParser
        from ..tokens import TextDelta as _TD

        events = ControlTokenParser().feed(text)
        spoken = "".join(e.text for e in events if isinstance(e, _TD)).strip()
        return self.estimator(spoken, language)

    @staticmethod
    def _fragments(seed: ScenarioSeed, rng: random.Random) -> list[str]:
        if seed.language == "zh":
            base = [
                "先梳理一下要点",
                "正在核对相关细节",
                "结论已经出来一半了",
                "还差最后一项验证",
            ]
        else:
            base = [
                "gathering the key facts",
                "checking the details now",
                "half of the answer is ready",
                "one last item to verify",
            ]
        k = rng.choice([2, 3])
        return base[:k]


class LLMDirector:
    """LLM-backed Director (paper's approach) with rule-based fallback."""

    def __init__(self, backend: LLMBackend, *, fallback: Director | None = None) -> None:
        self.backend = backend
        self.fallback: Director = fallback or RuleDirector()

    def annotate(self, script: str, seed: ScenarioSeed) -> AnnotatedDialog:
        prompt = (
            f"Language: {'Chinese' if seed.language == 'zh' else 'English'}\n"
            f"Target interaction patterns: {', '.join(p.value for p in seed.patterns)}\n"
            "Script to annotate:\n```\n" + script + "\n```\n"
        )
        raw = self.backend.generate(prompt, system=DIRECTOR_SYSTEM)
        utterances = _parse_llm_annotation(raw, seed.language)
        if utterances is None:
            return self.fallback.annotate(script, seed)
        return AnnotatedDialog(seed=seed, utterances=utterances)


def _parse_llm_annotation(raw: str, language: str
                          ) -> list[AnnotatedUtterance] | None:
    from ..tokens import ControlTokenParser, GhostText, ResultFragment, SharedSilence, TextDelta

    utterances: list[AnnotatedUtterance] = []
    ok = True
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("<") and line.endswith(">") and ">" not in line[1:-1]:
            utterances.append(
                AnnotatedUtterance(Speaker.ASSISTANT, line, kind=UtteranceKind.EVENT)
            )
            continue
        if line.startswith("[PEND") and line.endswith("S]"):
            utterances.append(
                AnnotatedUtterance(Speaker.ASSISTANT, line, kind=UtteranceKind.SILENCE)
            )
            continue
        if line[:3] in ("[U]", "[A]"):
            sp = Speaker.USER if line[:3] == "[U]" else Speaker.ASSISTANT
            body = line[3:].strip()
            events: list[Event] = ControlTokenParser().feed(body)
            ghost = any(isinstance(e, GhostText) for e in events)
            dur = 0.0
            if not ghost or any(isinstance(e, TextDelta) for e in events):
                spoken = "".join(e.text for e in events if isinstance(e, TextDelta))
                dur = estimate_duration(spoken, language=language)
            overlap = any(isinstance(e, OverlapOnset) for e in events)
            utterances.append(
                AnnotatedUtterance(
                    sp, body, duration_s=dur, overlap=overlap,
                    backchannel=is_backchannel(
                        "".join(e.text for e in events if isinstance(e, TextDelta))
                    ),
                )
            )
            continue
        ok = False  # unparseable line -> reject whole output
        break
    if not ok or not any(u.kind == UtteranceKind.EVENT or "[THINK]" in u.text for u in utterances):
        return None
    _ = (ResultFragment, SharedSilence)  # available for stricter future checks
    return utterances


def _script_turns(script: str) -> list[tuple[Speaker, str]]:
    """Structural parse of a writer script into (speaker, text) pairs."""
    events = parse_script(script)
    turns: list[tuple[Speaker, str]] = []
    cur_sp: Speaker | None = None
    cur: list[str] = []
    for ev in events:
        if isinstance(ev, TurnStart):
            if cur_sp is not None and cur:
                turns.append((cur_sp, "".join(cur).strip()))
            cur_sp = ev.speaker
            cur = []
        else:
            text = getattr(ev, "text", "")
            if text:
                cur.append(text)
    if cur_sp is not None and cur:
        turns.append((cur_sp, "".join(cur).strip()))
    return [(sp, t) for sp, t in turns if t]


_ = (Writer, parse_script_lines)  # imported for re-export convenience
