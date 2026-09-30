"""Asynchronous collaboration bridge between the two layers (paper 3.1).

Implements the paper's protocol on top of :mod:`asyncio`:

* the interaction layer calls :meth:`request` when its Thinker emits
  ``[THINK]`` — the call returns immediately (non-blocking) and the
  interaction layer keeps listening/speaking;
* the thinking layer's streamed fragments are wrapped as ``<...>`` token
  strings; the interaction layer drains them with :meth:`take_pending`
  every slice and injects them into its input stream;
* :meth:`halt` implements ``[WAIT]`` — cancels the in-flight request; any
  fragments not yet drained are dropped (aborted requests never inject).
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field

from .events import RuntimeEvent, ThinkingAborted, ThinkingFragment, ThinkingRequested
from .thinking import ThinkingContext, ThinkingLayer, ThinkingRequest


@dataclass
class _PendingRequest:
    request: ThinkingRequest
    task: asyncio.Task
    done: bool = False


@dataclass
class ThinkingBridge:
    """Non-blocking [THINK] / <...> / [WAIT] collaboration."""

    thinking: ThinkingLayer
    max_concurrent: int = 2
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1), repr=False)
    _inflight: list[_PendingRequest] = field(default_factory=list, repr=False)
    _pending_fragments: list[str] = field(default_factory=list, repr=False)
    events: list[RuntimeEvent] = field(default_factory=list)
    abort_count: int = 0

    # -- interaction-layer side ----------------------------------------------- #

    def _prune(self) -> None:
        self._inflight = [p for p in self._inflight if not p.task.done()]

    def request(self, context: ThinkingContext, query: str = "") -> str:
        """Dispatch a non-blocking reasoning request; returns its id."""
        self._prune()
        if len(self._inflight) >= self.max_concurrent:
            # paper semantics: supersede stale reasoning before new requests
            self.halt("capacity")
            self._inflight.clear()
        request_id = f"think-{next(self._counter)}"
        request = ThinkingRequest(request_id=request_id, context=context, query=query)

        async def _run() -> None:
            try:
                async for fragment in self.thinking.think(request):
                    self._pending_fragments.append(
                        f"<{fragment.strip()}>"
                    )
                    self.events.append(ThinkingFragment(text=fragment))
            except asyncio.CancelledError:
                raise
            finally:
                pending = next((p for p in self._inflight if p.request is request), None)
                if pending is not None:
                    pending.done = True

        self._inflight.append(
            _PendingRequest(
                request=request,
                task=asyncio.get_running_loop().create_task(_run()),
            )
        )
        self.events.append(ThinkingRequested(request_id=request_id, context_summary=context.summary()))
        return request_id

    def halt(self, reason: str = "user-interrupt") -> None:
        """[WAIT]: suspend/reset the current reasoning request(s)."""
        aborted = False
        for pending in self._inflight:
            if not pending.done and not pending.task.done():
                pending.task.cancel()
                aborted = True
        if aborted:
            self.abort_count += 1
            self.events.append(ThinkingAborted(reason=reason))
        # fragments of aborted requests must never be injected
        self._pending_fragments.clear()

    def take_pending(self) -> list[str]:
        """Drain wrapped ``<...>`` fragments for this slice."""
        out, self._pending_fragments = self._pending_fragments, []
        return out

    def collect_events(self) -> list[RuntimeEvent]:
        out, self.events = self.events, []
        return out

    # -- lifecycle ------------------------------------------------------------- #

    async def wait_idle(self) -> None:
        running = [p.task for p in self._inflight if not p.task.done()]
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    @property
    def busy(self) -> bool:
        return any(not p.task.done() for p in self._inflight)
