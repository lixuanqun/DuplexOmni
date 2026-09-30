"""Real-time metrics: RTF and latency statistics (paper target RTF < 1)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RTFMeter:
    """Real-Time Factor: compute seconds per slice / slice duration."""

    slice_s: float = 0.48
    samples: list[float] = field(default_factory=list)

    def measure(self, compute_s: float) -> float:
        rtf = compute_s / self.slice_s
        self.samples.append(rtf)
        return rtf

    @property
    def mean(self) -> float:
        return sum(self.samples) / max(len(self.samples), 1)

    @property
    def max(self) -> float:
        return max(self.samples, default=0.0)

    @property
    def ok(self) -> bool:
        return self.max < 1.0


@dataclass
class LatencyMeter:
    """Wall-clock latency of interaction milestones (paper: 0.506 s)."""

    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> float:
        self.marks[name] = time.monotonic()
        return self.marks[name]

    def since(self, start_name: str, end_name: str) -> float | None:
        if start_name in self.marks and end_name in self.marks:
            return self.marks[end_name] - self.marks[start_name]
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(int(q * (len(xs) - 1)), len(xs) - 1)
    return xs[idx]
