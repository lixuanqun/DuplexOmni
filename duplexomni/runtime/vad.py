"""Energy VAD for barge-in detection (runtime demo grade).

A production system plugs a streaming VAD/ASR front-end here; the interface
is deliberately tiny: feed 480 ms of PCM, get a speech decision.
"""

from __future__ import annotations

from ..codecs import rms
from ..config import RuntimeConfig


class EnergyVAD:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        cfg = config or RuntimeConfig()
        self.threshold = cfg.vad_threshold
        self.hangover_slices = max(1, round(cfg.vad_hangover_ms / cfg.slice_ms))
        self._hangover = 0

    def update(self, pcm: bytes) -> bool:
        """Returns True if the user is speaking in this slice."""
        if rms(pcm) > self.threshold:
            self._hangover = self.hangover_slices
            return True
        if self._hangover > 0:
            self._hangover -= 1
            return True
        return False

    def reset(self) -> None:
        self._hangover = 0
