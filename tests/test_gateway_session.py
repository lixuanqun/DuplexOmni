"""In-process full-duplex session: delegate, speak, barge-in cancel."""

from __future__ import annotations

import array
import asyncio
import os
import tempfile

from duplexomni.gateway.config import GatewayConfig
from duplexomni.gateway.engines.tts import ScriptedTts, decode_scripted, scripted_frames
from duplexomni.gateway.metrics import METERS
from duplexomni.gateway.protocol import decode_audio
from duplexomni.gateway.session import DuplexSession
from duplexomni.gateway.store import SessionStore
from duplexomni.gateway.tools import default_registry


def _pcm(level: float, n: int = 320) -> bytes:
    amp = int(max(-1.0, min(1.0, level)) * 32767)
    return array.array("h", [amp] * n).tobytes()


async def _collect(session: DuplexSession, pred, timeout: float = 2.0) -> list[dict]:
    messages: list[dict] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        left = deadline - loop.time()
        try:
            messages.append(await asyncio.wait_for(session.outbound.get(), timeout=min(0.05, left)))
        except TimeoutError:
            if pred(messages):
                return messages
    return messages


def test_delegate_calculate_streams_result():
    async def run() -> None:
        session = DuplexSession(GatewayConfig())
        await session.ingest_json({"type": "text", "text": "帮我算 12*(3+4)"})
        messages = await _collect(
            session,
            lambda msgs: any(m.get("status") == "done" for m in msgs),
        )
        blob = " ".join(str(m.get("text") or m.get("summary") or "") for m in messages)
        assert "84" in blob
        assert any(m.get("type") == "speak" for m in messages)
        assert any(m.get("type") == "delegation" for m in messages)
        await session.shutdown()

    asyncio.run(run())


def test_barge_in_cancels_slow_task_and_drops_late_result():
    async def slow(args, ctx):
        del args, ctx
        await asyncio.sleep(30)
        return "late"

    async def run() -> None:
        tools = default_registry()
        tools.register("slow", slow, timeout_s=10.0)
        session = DuplexSession(
            GatewayConfig(),
            tools=tools,
            planner=lambda goal: [("slow", {})],
        )
        await session.ingest_json({"type": "text", "text": "帮我算 1+1"})
        started = await _collect(
            session,
            lambda msgs: any(m.get("type") == "delegation" for m in msgs),
        )
        assert any(m.get("type") == "speak" for m in started)
        session.handle_frame(1, _pcm(0.25), False, False)
        after = await _collect(
            session,
            lambda msgs: any(m.get("status") == "cancelled" for m in msgs),
            timeout=1.0,
        )
        assert any(m.get("type") == "cut" for m in after)
        assert any(m.get("status") == "cancelled" for m in after)
        await asyncio.sleep(0.2)
        leftover = []
        while not session.outbound.empty():
            leftover.append(session.outbound.get_nowait())
        assert all("late" not in str(m) for m in leftover)
        await session.shutdown()

    asyncio.run(run())


def test_playback_echo_does_not_barge_in():
    async def run() -> None:
        session = DuplexSession(GatewayConfig())
        await session.ingest_json({"type": "text", "text": "你好"})
        await _collect(session, lambda msgs: any(m.get("type") == "speak" for m in msgs))
        assert session.floor.floor.value == "assistant"
        session.handle_frame(1, _pcm(0.02), True, True)
        assert session.floor.floor.value == "assistant"
        quiet = [m for m in _drain(session) if m.get("type") == "cut"]
        assert quiet == []
        await session.shutdown()

    asyncio.run(run())


def test_barge_in_stops_scripted_pcm():
    async def slow(args, ctx):
        del args, ctx
        await asyncio.sleep(30)
        return "late"

    async def run() -> None:
        tools = default_registry()
        tools.register("slow", slow, timeout_s=10.0)
        session = DuplexSession(
            GatewayConfig(),
            tools=tools,
            planner=lambda goal: [("slow", {})],
            tts=ScriptedTts(pace_s=0.05),
        )
        await session.ingest_json({"type": "text", "text": "帮我算 1+1"})
        first = await asyncio.wait_for(session.pcm_out.get(), 2)
        _index, pcm, _hint, _playback = decode_audio(session.live_pcm(first))
        assert decode_scripted(pcm) == (0, ord("好"))
        session.handle_frame(1, _pcm(0.4), False, False)
        after = await _collect(
            session,
            lambda msgs: any(m.get("status") == "cancelled" for m in msgs),
            timeout=1.0,
        )
        assert any(m.get("type") == "cut" for m in after)
        await asyncio.sleep(0.25)
        extra = []
        while not session.pcm_out.empty():
            live = session.live_pcm(session.pcm_out.get_nowait())
            if live is not None:
                extra.append(live)
        assert 1 + len(extra) < len(scripted_frames("好，我接着听，这件事交给后面做。"))
        await session.shutdown()

    asyncio.run(run())


def test_cut_invalidates_pcm_already_pulled_from_the_queue():
    async def run() -> None:
        session = DuplexSession(GatewayConfig(), tts=ScriptedTts(pace_s=0))
        await session.ingest_json({"type": "text", "text": "你好"})
        pulled = await asyncio.wait_for(session.pcm_out.get(), 2)
        assert session.live_pcm(pulled) is not None
        session.handle_frame(1, _pcm(0.4), False, False)
        assert session.live_pcm(pulled) is None
        await session.shutdown()

    asyncio.run(run())


def test_adapted_echo_does_not_barge_but_new_speech_does():
    async def run() -> None:
        session = DuplexSession(GatewayConfig(), tts=ScriptedTts(pace_s=0))
        await session.ingest_json({"type": "text", "text": "你好"})
        await _collect(session, lambda msgs: any(m.get("type") == "speak" for m in msgs))
        echo = _pcm(0.2)
        for _ in range(12):
            session.aec.push_far(echo)
            session.aec.cancel(echo)
        session.handle_frame(1, echo, False, True)
        assert session.floor.floor.value == "assistant"
        session.handle_frame(2, _pcm(0.45), False, True)
        assert any(item.get("type") == "cut" for item in _drain(session))
        await session.shutdown()

    asyncio.run(run())


def test_turn_latency_covers_speech_and_task():
    async def run() -> None:
        METERS.reset()
        session = DuplexSession(GatewayConfig(), tts=ScriptedTts(pace_s=0))
        await session.ingest_json({"type": "text", "text": "帮我算 12*(3+4)"})
        await _collect(session, lambda msgs: any(m.get("status") == "done" for m in msgs))
        text = METERS.render()
        assert "duplex_reply_ms_count 1" in text
        assert "duplex_first_audio_ms_count 1" in text
        assert "duplex_task_ms_count 1" in text
        await session.shutdown()

    asyncio.run(run())


def test_restart_fails_a_task_left_running():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        store = SessionStore(path)
        store.ensure_session("sess-restart")
        store.upsert_task("sess-restart", "task-open", status="running", goal="算一下")
        store.close()
        reopened = SessionStore(path)
        rows = reopened.list_tasks("sess-restart")
        assert rows[0]["status"] == "failed"
        assert "重启" in rows[0]["detail"]
        reopened.close()
    finally:
        os.remove(path)


def test_transcript_is_replayed_without_speaking_again():
    async def run() -> None:
        store = SessionStore(":memory:")
        session = DuplexSession(GatewayConfig(), store=store, session_id="sess-turns", tts=ScriptedTts(pace_s=0))
        store.ensure_session(session.session_id)
        await session.ingest_json({"type": "text", "text": "帮我算 12*(3+4)"})
        await _collect(session, lambda msgs: any(m.get("status") == "done" for m in msgs))
        await session.shutdown()
        turns = store.list_turns(session.session_id)
        assert any(item["role"] == "user" and "12*(3+4)" in item["text"] for item in turns)
        assert any(item["role"] == "assistant" for item in turns)

        resumed = DuplexSession(GatewayConfig(), store=store, session_id="sess-turns", tts=ScriptedTts(pace_s=0))
        resumed.replay()
        replayed = _drain(resumed)
        history = [item for item in replayed if item.get("type") == "history"]
        assert any(item.get("role") == "user" for item in history)
        assert not any(item.get("type") == "speak" for item in replayed)
        await resumed.shutdown()
        store.close()

    asyncio.run(run())


def test_delegated_result_is_stored_and_replayed():
    async def run() -> None:
        store = SessionStore(":memory:")
        session = DuplexSession(GatewayConfig(), store=store, session_id="sess-persist")
        store.ensure_session(session.session_id)
        await session.ingest_json({"type": "text", "text": "帮我算 12*(3+4)"})
        await _collect(session, lambda msgs: any(m.get("status") == "done" for m in msgs))
        rows = store.list_tasks(session.session_id)
        assert rows and rows[-1]["status"] == "done"
        assert "84" in rows[-1]["summary"]
        await session.shutdown()

        resumed = DuplexSession(GatewayConfig(), store=store, session_id="sess-persist")
        resumed.replay()
        replayed = _drain(resumed)
        assert any("84" in str(item.get("summary") or "") for item in replayed)
        await resumed.shutdown()
        store.close()

    asyncio.run(run())


def test_configured_tool_module_is_callable():
    session = DuplexSession(GatewayConfig(tool_modules="ping"))
    assert session.tools.known("ping")


def _drain(session: DuplexSession) -> list[dict]:
    found = []
    while not session.outbound.empty():
        found.append(session.outbound.get_nowait())
    return found
