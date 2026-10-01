"""Streaming ASR port.

``push`` consumes 20 ms PCM and may return partials or a final. ``feed_text``
is the same final path for keyboard input and for tests that already have a
transcript. The default engine does not decode audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AsrEvent:
    text: str
    final: bool


class AsrPort(Protocol):
    def push(self, pcm: bytes) -> list[AsrEvent]:
        """Decode one uplink frame. Empty means no new transcript."""

    def feed_text(self, text: str, *, final: bool) -> list[AsrEvent]:
        """Accept text that did not come from audio."""


class ScriptedAsr:
    """No acoustic model. Audio is ignored; text is passed through."""

    def push(self, pcm: bytes) -> list[AsrEvent]:
        del pcm
        return []

    def feed_text(self, text: str, *, final: bool) -> list[AsrEvent]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        return [AsrEvent(cleaned, final)]
