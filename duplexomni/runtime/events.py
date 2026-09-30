"""Runtime events emitted by the interaction layer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeEvent:
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class AssistantTextDelta(RuntimeEvent):
    text: str = ""


@dataclass
class AssistantSpeechSlice(RuntimeEvent):
    index: int = 0
    pcm: bytes = b""
    codes: list = field(default_factory=list)
    muted: bool = False  # ghost/silence slice: caches advanced, no audio


@dataclass
class ThinkingRequested(RuntimeEvent):
    request_id: str = ""
    context_summary: str = ""


@dataclass
class ThinkingFragment(RuntimeEvent):
    text: str = ""


@dataclass
class ThinkingAborted(RuntimeEvent):
    reason: str = ""


@dataclass
class BargeInDetected(RuntimeEvent):
    slice_index: int = 0


@dataclass
class SpeechCut(RuntimeEvent):
    ghost_text: str = ""


@dataclass
class SharedSilence(RuntimeEvent):
    seconds: float = 0.0


@dataclass
class Overlap(RuntimeEvent):
    """User speech detected while the assistant is speaking."""

    slice_index: int = 0


@dataclass
class SliceOutput:
    """Everything the interaction layer produced for one 480 ms slice."""

    index: int
    text: str = ""
    pcm: bytes = b""
    codes: list = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)
    compute_s: float = 0.0
    rtf: float = 0.0
    user_speaking: bool = False
    assistant_speaking: bool = False

    def events_of(self, kind: type) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def summarise(outputs: list[SliceOutput]) -> dict[str, Any]:
    """Aggregate a finished session into headline metrics."""
    rtfs = [o.rtf for o in outputs if o.rtf > 0]
    return {
        "slices": len(outputs),
        "duration_s": len(outputs) * 0.48,
        "mean_rtf": sum(rtfs) / max(len(rtfs), 1),
        "max_rtf": max(rtfs, default=0.0),
        "barge_ins": sum(len(o.events_of(BargeInDetected)) for o in outputs),
        "thinking_requests": sum(len(o.events_of(ThinkingRequested)) for o in outputs),
        "thinking_fragments": sum(len(o.events_of(ThinkingFragment)) for o in outputs),
        "thinking_aborts": sum(len(o.events_of(ThinkingAborted)) for o in outputs),
        "speech_cuts": sum(len(o.events_of(SpeechCut)) for o in outputs),
        "shared_silences": sum(len(o.events_of(SharedSilence)) for o in outputs),
        "text": "".join(o.text for o in outputs),
        "rtf_ok": all(r < 1.0 for r in rtfs),
    }
