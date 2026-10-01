"""Energy VAD adapter.

The session depends on ``update`` plus an ``endpoint`` edge. A neural VAD
can replace this class without touching the floor machine.
"""

from __future__ import annotations

from ..vad import AdaptiveVAD


class EnergyVad(AdaptiveVAD):
    """CPU energy gate. Same behaviour as :class:`AdaptiveVAD`."""
