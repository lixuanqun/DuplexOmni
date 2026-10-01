"""Streaming TTS port.

``synthesize`` yields 20 ms 16-bit mono PCM and must stop when the caller
cancels the task. The scripted engine writes a marker into each frame so
tests can see how far playback got.
"""

from __future__ import annotations

import array
import asyncio
import math
from collections.abc import AsyncIterator
from typing import Protocol

MARKER = 0x51A1
_MAX_FRAMES = 16


class TtsPort(Protocol):
    def synthesize(self, text: str, turn_id: int) -> AsyncIterator[bytes]:
        """Yield one PCM frame per 20 ms. Cancellation ends the stream."""


def scripted_frames(text: str, *, samples: int = 320) -> list[bytes]:
    """One 20 ms frame per character.

    Samples 0–2 stay a marker so tests can see which character was reached.
    The rest of the frame is a tone at a pitch derived from that character,
    loud enough to hear on headphones and still under full scale.
    """
    chars = list(text[:_MAX_FRAMES]) or [" "]
    rate = samples / 0.020
    frames: list[bytes] = []
    for index, char in enumerate(chars):
        buf = array.array("h", [0] * samples)
        code = min(ord(char), 0x7FFF)
        buf[0] = MARKER
        buf[1] = index
        buf[2] = code
        freq = 220.0 + (code % 40) * 15.0
        amp = 8000.0
        for n in range(3, samples):
            phase = 2.0 * math.pi * freq * ((index * samples) + n) / rate
            buf[n] = int(amp * math.sin(phase))
        frames.append(buf.tobytes())
    return frames


def decode_scripted(pcm: bytes) -> tuple[int, int] | None:
    """Return ``(frame_index, code_unit)`` when ``pcm`` is a scripted frame."""
    usable = len(pcm) - (len(pcm) % 2)
    if usable < 6:
        return None
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    if samples[0] != MARKER:
        return None
    return int(samples[1]), int(samples[2])


class ScriptedTts:
    def __init__(self, *, sample_rate: int = 16_000, frame_ms: int = 20, pace_s: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.pace_s = pace_s

    async def synthesize(self, text: str, turn_id: int) -> AsyncIterator[bytes]:
        del turn_id
        samples = int(self.sample_rate * self.frame_ms / 1000)
        for frame in scripted_frames(text, samples=max(samples, 1)):
            yield frame
            await asyncio.sleep(self.pace_s)
