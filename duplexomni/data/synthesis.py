"""TTS synthesis backends (paper Sec. 4.1: Qwen3-TTS).

The Director annotates before audio exists, so duration *estimation* lives
here as the single source of truth; :class:`MockTTS` also renders audible
deterministic tone speech so demos produce real WAV output without any
external service.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Protocol

from ..config import SAMPLE_RATE_HZ

__all__ = ["TTSBackend", "MockTTS", "UtteranceAudio", "estimate_duration"]

# chars per second, per language (spoken-rate heuristic)
_CHARS_PER_SECOND = {"zh": 4.8, "en": 13.5}
_MIN_DURATION_S = 0.4
_BACKCHANNEL_MAX_S = 0.6


@dataclass
class UtteranceAudio:
    text: str
    duration_s: float
    pcm: bytes = b""  # 16-bit mono LE, may be empty when only estimating


class TTSBackend(Protocol):
    def synthesize(self, text: str, *, speaker: str, language: str = "en") -> UtteranceAudio: ...


def estimate_duration(text: str, *, language: str = "en") -> float:
    """Estimate spoken duration of ``text`` (control tokens excluded by caller)."""
    rate = _CHARS_PER_SECOND.get(language, _CHARS_PER_SECOND["en"])
    n = len([c for c in text if not c.isspace()])
    return max(_MIN_DURATION_S, n / rate)


class MockTTS:
    """Offline deterministic TTS: duration heuristic + audible tone speech.

    The waveform is a sum of harmonics with a syllable-rate envelope, seeded
    by the text, so the same text always renders the same audio.  Speaker
    changes the base pitch (user 170 Hz, assistant 225 Hz) and stereo pan is
    not used (mono).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE_HZ) -> None:
        self.sample_rate = sample_rate

    def synthesize(self, text: str, *, speaker: str, language: str = "en") -> UtteranceAudio:
        duration = estimate_duration(text, language=language)
        return UtteranceAudio(text=text, duration_s=duration, pcm=self._render(text, duration, speaker))

    # -- rendering ------------------------------------------------------------ #

    def _render(self, text: str, duration_s: float, speaker: str) -> bytes:
        n_samples = int(duration_s * self.sample_rate)
        if n_samples <= 0:
            return b""
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        base_f0 = 170.0 if speaker == "user" else 225.0
        syllable_hz = 4.2 if any("\u4e00" <= c <= "\u9fff" for c in text) else 5.5
        try:  # numpy fast path (kept optional; stdlib fallback below)
            import numpy as np

            t = np.arange(n_samples, dtype=np.float64) / self.sample_rate
            syl = 2 * math.pi * syllable_hz * t
            env = 0.55 + 0.45 * np.sin(syl + (seed % 7))
            jitter = 1.0 + 0.04 * np.sin(2 * math.pi * 0.7 * t + (seed % 11))
            f0 = base_f0 * jitter
            val = 5200 * np.maximum(env, 0.05) * (
                np.sin(2 * np.pi * f0 * t)
                + 0.4 * np.sin(2 * math.pi * 2 * f0 * t + (seed % 5))
                + 0.15 * np.sin(2 * math.pi * 3 * f0 * t + (seed % 3))
            )
            pcm = np.clip(val, -32000, 32000).astype("<i2")
            return pcm.tobytes()
        except ImportError:
            values = bytearray()
            for n in range(n_samples):
                t = n / self.sample_rate
                syl = 2 * math.pi * syllable_hz * t
                env = 0.55 + 0.45 * math.sin(syl + (seed % 7))
                jitter = 1.0 + 0.04 * math.sin(2 * math.pi * 0.7 * t + (seed % 11))
                f0 = base_f0 * jitter
                val = 5200 * max(env, 0.05) * (
                    math.sin(2 * math.pi * f0 * t)
                    + 0.4 * math.sin(2 * math.pi * 2 * f0 * t + (seed % 5))
                    + 0.15 * math.sin(2 * math.pi * 3 * f0 * t + (seed % 3))
                )
                values += struct.pack("<h", int(max(-32000, min(32000, val))))
            return bytes(values)


def is_backchannel(text: str) -> bool:
    """Short acknowledgement utterance ("嗯嗯", "yeah") that overlaps speech."""
    stripped = "".join(c for c in text if not c.isspace() and c not in "ˆ^")
    return 0 < len(stripped) <= 4 and estimate_duration(stripped) <= _BACKCHANNEL_MAX_S
