"""Adaptive energy VAD for 20 ms frames.

A neural VAD can replace :meth:`AdaptiveVAD.update` later. The contract is
one boolean per frame plus an ``endpoint`` edge when speech ends, after a
hangover so a short pause does not close the user's turn.
"""

from __future__ import annotations

import array
import math


def rms(pcm: bytes) -> float:
    """RMS of 16-bit mono PCM, normalised to [0, 1]."""
    if not pcm:
        return 0.0
    usable = len(pcm) - (len(pcm) % 2)
    if usable == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    acc = 0
    for sample in samples:
        acc += sample * sample
    return math.sqrt(acc / len(samples)) / 32768.0


class AdaptiveVAD:
    def __init__(
        self,
        *,
        frame_ms: int = 20,
        threshold: float = 0.015,
        hangover_ms: int = 320,
        noise_alpha: float = 0.95,
    ) -> None:
        self.threshold = threshold
        self.noise_alpha = noise_alpha
        self.noise_rms = 0.005
        self.hangover_frames = max(1, hangover_ms // max(frame_ms, 1))
        self._hangover = 0
        self._speaking = False
        self.endpoint = False

    def update(self, pcm: bytes) -> bool:
        level = rms(pcm)
        self.endpoint = False
        if level < self.noise_rms * 2.5:
            self.noise_rms = self.noise_alpha * self.noise_rms + (1.0 - self.noise_alpha) * level
        gate = max(self.threshold, self.noise_rms * 3.5)
        if level > gate:
            self._hangover = self.hangover_frames
            speaking = True
        elif self._hangover > 0:
            self._hangover -= 1
            speaking = True
        else:
            speaking = False
        if self._speaking and not speaking:
            self.endpoint = True
        self._speaking = speaking
        return speaking

    def reset(self) -> None:
        self._hangover = 0
        self._speaking = False
        self.endpoint = False
        self.noise_rms = 0.005
