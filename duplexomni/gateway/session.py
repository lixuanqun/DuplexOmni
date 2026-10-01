"""One full-duplex session: audio floor, fast speech, delegated work.

The socket owner calls :meth:`ingest_json` and :meth:`ingest_audio`. A
background loop drains audio and releases the assistant floor on its hold
timer. Nothing in that loop awaits a tool or an LLM.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable

from .config import GatewayConfig
from .dialogue import estimate_hold_s, fast_ack
from .engines import build_asr, build_tts, build_vad
from .floor import Floor, FloorController
from .metrics import METERS
from .protocol import encode_audio
from .router import route_utterance
from .store import SessionStore
from .tools import Memory, ToolRegistry, default_registry
from .vad import rms
from .worker import TaskSpec, execute_task

log = logging.getLogger(__name__)

_CRITICAL = frozenset({"cut", "speak", "delegation", "task", "error", "hello", "floor", "fragment"})


class DuplexSession:
    def __init__(
        self,
        config: GatewayConfig | None = None,
        *,
        tools: ToolRegistry | None = None,
        planner: Callable | None = None,
        llm: object | None = None,
        session_id: str | None = None,
        vad: object | None = None,
        asr: object | None = None,
        tts: object | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.config = config or GatewayConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.floor = FloorController()
        self.vad = vad or build_vad(self.config.vad, frame_ms=self.config.frame_ms)
        self.asr = asr or build_asr(self.config.asr)
        self.tts = tts or build_tts(
            self.config.tts,
            sample_rate=self.config.sample_rate,
            frame_ms=self.config.frame_ms,
        )
        self.store = store
        self.tools = tools or default_registry()
        self.planner = planner
        self.llm = llm
        self.memory = Memory()
        self.outbound: asyncio.Queue[dict] = asyncio.Queue(maxsize=self.config.outbound_queue)
        self.pcm_out: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=self.config.outbound_queue)
        self.audio_q: asyncio.Queue[tuple[int, bytes, bool, bool]] = asyncio.Queue(
            maxsize=self.config.audio_queue
        )
        self._speech_q: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue(maxsize=8)
        self._tts_task: asyncio.Task | None = None
        self._pcm_index = 0
        self.pcm_epoch = 0
        self._deferred: list[str] = []
        self._partial = ""
        self._generation = 0
        self._reply_task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._worker: asyncio.Task | None = None
        self._worker_cancel: asyncio.Event | None = None
        self._runner: asyncio.Task | None = None
        self._closed = False
        self._bad_messages = 0
        self._last_text = ""
        self._last_text_at = 0.0
        self._last_rx = time.monotonic()
        self._asked_at: dict[int, float] = {}

    def hello_message(self) -> dict:
        return {
            "type": "hello",
            "session_id": self.session_id,
            "sample_rate": self.config.sample_rate,
            "frame_ms": self.config.frame_ms,
            "slice_ms": 480,
            "llm": self.llm is not None,
            "asr": self.config.asr,
            "tts": self.config.tts,
            "vad": self.config.vad,
            "resumed": bool(self.store and self.store.list_tasks(self.session_id)),
        }

    def replay(self) -> None:
        """Re-send stored tasks after a client reconnects with the same id."""
        if self.store is None:
            return
        for task in self.store.list_tasks(self.session_id):
            payload = {
                "type": "task",
                "task_id": task["task_id"],
                "status": task["status"],
                "goal": task["goal"],
                "tool": task["tool"],
                "summary": task["summary"],
                "detail": task["detail"],
            }
            self.emit({key: value for key, value in payload.items() if value})

    def emit(self, message: dict) -> None:
        try:
            self.outbound.put_nowait(message)
            return
        except asyncio.QueueFull:
            pass
        if message.get("type") not in _CRITICAL:
            METERS.inc("outbound_dropped")
            return
        kept: list[dict] = []
        dropped = False
        while True:
            try:
                item = self.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not dropped and item.get("type") not in _CRITICAL:
                dropped = True
                METERS.inc("outbound_dropped")
                continue
            kept.append(item)
        for item in kept:
            try:
                self.outbound.put_nowait(item)
            except asyncio.QueueFull:
                METERS.inc("outbound_dropped")
        try:
            self.outbound.put_nowait(message)
        except asyncio.QueueFull:
            METERS.inc("outbound_dropped")

    def emit_pcm(self, pcm: bytes) -> None:
        self._pcm_index = (self._pcm_index + 1) & 0xFFFFFFFF
        frame = (self.pcm_epoch, encode_audio(self._pcm_index, pcm))
        try:
            self.pcm_out.put_nowait(frame)
            return
        except asyncio.QueueFull:
            pass
        try:
            self.pcm_out.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            self.pcm_out.put_nowait(frame)
        except asyncio.QueueFull:
            METERS.inc("outbound_dropped")

    def live_pcm(self, item: tuple[int, bytes]) -> bytes | None:
        """Drop audio that was encoded before the latest cut."""
        epoch, frame = item
        if epoch != self.pcm_epoch:
            return None
        return frame

    def _drop_pcm(self) -> None:
        self.pcm_epoch += 1
        while True:
            try:
                self.pcm_out.get_nowait()
            except asyncio.QueueEmpty:
                return

    def touch(self) -> None:
        self._last_rx = time.monotonic()

    async def ingest_json(self, message: dict) -> None:
        self.touch()
        kind = message.get("type")
        if kind == "text":
            await self._accept_text(str(message.get("text") or ""), final=True)
        elif kind == "transcript":
            await self._accept_text(str(message.get("text") or ""), final=bool(message.get("final")))
        elif kind == "barge":
            if self.floor.floor == Floor.ASSISTANT:
                self._invalidate("barge-in")
                self._apply_all(self.floor.user_voice(now=time.monotonic()))
        elif kind == "cancel":
            self._invalidate("client")
        elif kind == "playback":
            self._apply_all(
                self.floor.note_playback(message.get("state") == "start", now=time.monotonic())
            )
        elif kind == "ping":
            self.emit({"type": "pong", "t": message.get("t")})
        elif kind in {"hello", "bye"}:
            return
        else:
            self._bad_messages += 1

    def ingest_audio(self, index: int, pcm: bytes, speech_hint: bool, playback: bool) -> None:
        self.touch()
        item = (index, pcm, speech_hint, playback)
        try:
            self.audio_q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self.audio_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.audio_q.put_nowait(item)
            except asyncio.QueueFull:
                pass
            METERS.inc("audio_dropped")

    def handle_frame(self, index: int, pcm: bytes, speech_hint: bool, playback: bool) -> None:
        del index
        level = rms(pcm)
        now = time.monotonic()
        server_play = self._tts_live() or self.floor.server_playing(now)
        echo = (server_play or playback) and level < self.config.echo_rms
        if echo:
            self.vad.update(b"\x00" * max(len(pcm), 2))
            speaking = False
            heard = b""
        else:
            speaking = self.vad.update(pcm) or (speech_hint and level >= self.config.hint_rms)
            heard = pcm
        if speaking and self.floor.floor == Floor.ASSISTANT:
            METERS.inc("barge_ins")
            self._invalidate("barge-in")
        if speaking:
            self._apply_all(self.floor.user_voice(now=now))
        for event in self.asr.push(heard):
            self._on_asr(event)
        if self.vad.endpoint:
            self._apply_all(self.floor.user_release(now=now))
            self._flush_deferred()
            if self._partial.strip():
                text = self._partial.strip()
                self._partial = ""
                self._kick(text)

    async def run(self) -> None:
        period = self.config.frame_ms / 1000
        while not self._closed:
            idle_left = self.config.idle_timeout_s - (time.monotonic() - self._last_rx)
            if idle_left <= 0:
                self.emit({"type": "error", "code": "idle", "message": "会话空闲超时"})
                self._closed = True
                break
            try:
                index, pcm, hint, playback = await asyncio.wait_for(self.audio_q.get(), timeout=period)
            except TimeoutError:
                self._apply_all(self.floor.tick(now=time.monotonic()))
                continue
            try:
                self.handle_frame(index, pcm, hint, playback)
                self._apply_all(self.floor.tick(now=time.monotonic()))
            except Exception:
                log.exception("session %s frame failed", self.session_id)
                METERS.inc("frame_errors")

    def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self.run(), name=f"duplex-{self.session_id}")

    async def shutdown(self) -> None:
        if self._closed:
            return
        worker = self._worker
        self._closed = True
        self._invalidate("session-end")
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await asyncio.wait_for(worker, timeout=1)
            except (asyncio.CancelledError, TimeoutError):
                pass
        if self._runner is not None and not self._runner.done():
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        for task in list(self._bg):
            task.cancel()
        if self._bg:
            await asyncio.gather(*self._bg, return_exceptions=True)
        if self.store is not None:
            self.store.interrupt_running(self.session_id)

    def _tts_live(self) -> bool:
        return self._tts_task is not None and not self._tts_task.done()

    def _on_asr(self, event) -> None:
        text = event.text.strip()[: self.config.max_text]
        if not text:
            return
        if event.final:
            self._partial = ""
            self._kick(text)
        else:
            self._partial = text
            self.emit({"type": "transcript", "role": "user", "text": text, "final": False})

    async def _accept_text(self, text: str, *, final: bool) -> None:
        for event in self.asr.feed_text(text[: self.config.max_text], final=final):
            if event.final:
                self._partial = ""
                await self._schedule(event.text[: self.config.max_text])
            else:
                self._partial = event.text[: self.config.max_text]

    async def _schedule(self, text: str) -> None:
        text = " ".join(text.split())[: self.config.max_text]
        if not text or self._closed:
            return
        now = time.monotonic()
        if text == self._last_text and now - self._last_text_at < 0.35:
            return
        self._last_text = text
        self._last_text_at = now
        self._invalidate("new-utterance")
        generation = self._generation
        self._asked_at[generation] = now
        self.emit({"type": "transcript", "role": "user", "text": text, "final": True})
        self._reply_task = self._spawn(self._reply(text, generation))

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    def _kick(self, text: str) -> None:
        self._spawn(self._schedule(text))

    def _invalidate(self, reason: str) -> None:
        self._generation += 1
        self._deferred.clear()
        self._cancel_tts()
        task = self._reply_task
        self._reply_task = None
        if task and not task.done():
            task.cancel()
        self._apply_all(self.floor.preempt(reason))

    def _cancel_tts(self) -> None:
        self._drop_pcm()
        while True:
            try:
                self._speech_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        task = self._tts_task
        self._tts_task = None
        if task is not None and not task.done():
            task.cancel()

    def _enqueue_speech(self, text: str, turn_id: int) -> None:
        item = (text, turn_id, self._generation)
        try:
            self._speech_q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._speech_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._speech_q.put_nowait(item)
            except asyncio.QueueFull:
                return
        if not self._tts_live():
            self._tts_task = self._spawn(self._tts_loop())

    async def _tts_loop(self) -> None:
        frame_s = self.config.frame_ms / 1000
        try:
            while not self._closed:
                text, turn_id, generation = await self._speech_q.get()
                if generation != self._generation:
                    continue
                async for pcm in self.tts.synthesize(text, turn_id):
                    if generation != self._generation or self._closed:
                        return
                    if self.floor.floor != Floor.ASSISTANT:
                        return
                    self.emit_pcm(pcm)
                    self.floor.note_sent_frame(now=time.monotonic(), frame_s=frame_s)
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            return

    async def _reply(self, text: str, generation: int) -> None:
        try:
            route = route_utterance(text)
            ack = fast_ack(text, route)
            if route == "chat" and self.llm is not None:
                try:
                    generated = await asyncio.wait_for(
                        self.llm.short_reply(text),  # type: ignore[attr-defined]
                        timeout=self.config.fast_timeout_s,
                    )
                    if generated.strip():
                        ack = generated.strip()[:200]
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning("session %s fast model failed", self.session_id)
                    METERS.inc("fast_model_errors")
            if generation != self._generation or self._closed:
                return
            asked = self._asked_at.pop(generation, None)
            if asked is not None:
                METERS.observe("reply_ms", (time.monotonic() - asked) * 1000)
            self._apply_all(self.floor.begin_speech(ack, estimate_hold_s(ack), now=time.monotonic()))
            if route == "delegate":
                self._start_task(text, generation)
        except asyncio.CancelledError:
            return

    def _start_task(self, goal: str, generation: int) -> None:
        if generation != self._generation or self._closed:
            return
        task_id = uuid.uuid4().hex[:12]
        self._pending_goal = goal
        self._apply_all(self.floor.start_thinking(task_id))
        spec = TaskSpec(task_id=task_id, goal=goal, session_id=self.session_id)
        self._spawn_worker(spec, generation)
        METERS.inc("delegations")

    def _spawn_worker(self, spec: TaskSpec, generation: int) -> None:
        self._stop_worker("superseded")
        cancel = asyncio.Event()
        self._worker_cancel = cancel
        self._worker = asyncio.create_task(
            self._run_worker(spec, cancel, generation),
            name=f"duplex-task-{spec.task_id}",
        )

    async def _run_worker(self, spec: TaskSpec, cancel: asyncio.Event, generation: int) -> None:
        async def on_event(event: dict) -> None:
            await self._on_worker(event, spec.task_id, generation)

        try:
            kwargs = {
                "tools": self.tools,
                "memory": self.memory,
                "on_event": on_event,
                "cancel": cancel,
                "task_timeout_s": self.config.task_timeout_s,
                "llm": self.llm,
            }
            if self.planner is not None:
                kwargs["planner"] = self.planner
            await execute_task(spec, **kwargs)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("session %s task %s failed", self.session_id, spec.task_id)
            if generation == self._generation:
                self.emit({
                    "type": "task",
                    "task_id": spec.task_id,
                    "status": "failed",
                    "detail": "任务执行失败",
                })
                self._apply_all(self.floor.finish_thinking(spec.task_id))

    async def _on_worker(self, event: dict, task_id: str, generation: int) -> None:
        if generation != self._generation or self._closed:
            return
        kind = event.get("kind")
        if kind == "fragment":
            text = str(event.get("text") or "")[:500]
            if not text:
                return
            self._save_task(task_id, status="running", detail=text)
            self.emit({"type": "fragment", "task_id": task_id, "text": text})
            self._apply_all(
                self.floor.extend_speech(text, estimate_hold_s(text), now=time.monotonic())
            )
            return
        if kind != "status":
            return
        status = event.get("status")
        payload = {
            "type": "task",
            "task_id": task_id,
            "status": status,
            "tool": event.get("tool"),
            "summary": event.get("summary"),
            "detail": event.get("detail"),
            "goal": event.get("goal"),
        }
        self.emit({k: v for k, v in payload.items() if v is not None})
        self._save_task(
            task_id,
            status=str(status or "running"),
            tool=str(event.get("tool") or ""),
            summary=str(event.get("summary") or ""),
            detail=str(event.get("detail") or ""),
            goal=str(event.get("goal") or ""),
        )
        if status in {"done", "failed"}:
            self._apply_all(self.floor.finish_thinking(task_id))

    def _stop_worker(self, reason: str) -> None:
        del reason
        cancel = self._worker_cancel
        worker = self._worker
        self._worker_cancel = None
        self._worker = None
        if cancel is not None:
            cancel.set()
        if worker is not None and not worker.done():
            worker.cancel()
            METERS.inc("task_cancels")

    def _apply_all(self, commands: list) -> None:
        for command in commands:
            self._apply(command)

    def _apply(self, command) -> None:
        name = command.name
        data = command.data
        if name == "speak":
            self.emit({
                "type": "speak",
                "text": data["text"],
                "turn_id": data["turn_id"],
                "hold_s": round(float(data.get("hold_s", 0.0)), 3),
            })
            self._enqueue_speech(str(data["text"]), int(data["turn_id"]))
        elif name == "defer_speech":
            self._deferred.append(data["text"])
        elif name == "cut":
            self._deferred.clear()
            self._drop_pcm()
            self.emit({"type": "cut", "reason": data.get("reason", "")})
        elif name == "cancel_task":
            self._stop_worker(str(data.get("reason") or "cancel"))
            self._deferred.clear()
            task_id = data.get("task_id")
            reason = str(data.get("reason") or "")
            status = "interrupted" if reason == "session-end" else "cancelled"
            if task_id:
                self._save_task(str(task_id), status=status, detail=reason)
            self.emit({
                "type": "task",
                "status": status,
                "reason": reason,
                **({"task_id": task_id} if task_id else {}),
            })
        elif name == "start_task":
            goal = getattr(self, "_pending_goal", "")
            self._save_task(str(data["task_id"]), status="running", goal=str(goal))
            self.emit({
                "type": "delegation",
                "task_id": data["task_id"],
                "status": "running",
                "goal": goal,
            })
        elif name == "floor":
            self.emit({"type": "floor", "floor": data.get("floor"), "turn_id": data.get("turn_id")})
        elif name == "task_idle":
            return

    def _save_task(
        self,
        task_id: str,
        *,
        status: str,
        goal: str = "",
        tool: str = "",
        summary: str = "",
        detail: str = "",
    ) -> None:
        if self.store is None or not task_id:
            return
        self.store.upsert_task(
            self.session_id,
            task_id,
            status=status,
            goal=goal,
            tool=tool,
            summary=summary,
            detail=detail,
        )

    def _flush_deferred(self) -> None:
        if self.floor.floor == Floor.USER or not self._deferred:
            return
        pending = self._deferred
        self._deferred = []
        now = time.monotonic()
        for text in pending:
            self._apply_all(self.floor.begin_speech(text, estimate_hold_s(text), now=now))
