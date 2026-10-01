"""In-process counters and a Prometheus text exposition.

The gateway does not take a metrics library dependency. Scrapes of
``/metrics`` are the operational surface.
"""

from __future__ import annotations

from collections import defaultdict

BUCKETS_MS = (5, 10, 20, 50, 100, 200, 480, 1000, 2500, 5000)


class Meters:
    def __init__(self) -> None:
        self.counters: dict[str, int] = defaultdict(int)
        self._hist_buckets: dict[str, list[int]] = defaultdict(lambda: [0] * (len(BUCKETS_MS) + 1))
        self._hist_sum: dict[str, float] = defaultdict(float)
        self._hist_count: dict[str, int] = defaultdict(int)

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def set_gauge(self, name: str, value: int) -> None:
        self.counters[name] = value

    def observe(self, name: str, value_ms: float) -> None:
        buckets = self._hist_buckets[name]
        placed = False
        for i, edge in enumerate(BUCKETS_MS):
            if value_ms <= edge:
                buckets[i] += 1
                placed = True
                break
        if not placed:
            buckets[-1] += 1
        self._hist_sum[name] += value_ms
        self._hist_count[name] += 1

    def reset(self) -> None:
        self.counters.clear()
        self._hist_buckets.clear()
        self._hist_sum.clear()
        self._hist_count.clear()

    def render(self) -> str:
        lines: list[str] = []
        for name in sorted(self.counters):
            lines.append(f"# TYPE duplex_{name} gauge")
            lines.append(f"duplex_{name} {self.counters[name]}")
        for name in sorted(self._hist_count):
            lines.append(f"# TYPE duplex_{name} histogram")
            running = 0
            buckets = self._hist_buckets[name]
            for edge, count in zip(BUCKETS_MS, buckets, strict=False):
                running += count
                lines.append(f'duplex_{name}_bucket{{le="{edge}"}} {running}')
            running += buckets[-1]
            lines.append(f'duplex_{name}_bucket{{le="+Inf"}} {running}')
            lines.append(f"duplex_{name}_sum {self._hist_sum[name]}")
            lines.append(f"duplex_{name}_count {self._hist_count[name]}")
        lines.append("")
        return "\n".join(lines)


METERS = Meters()
