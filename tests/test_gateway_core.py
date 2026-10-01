"""Floor machine, routing, and safe calculation."""

from __future__ import annotations

import pytest

from duplexomni.gateway.engines.tts import decode_scripted, scripted_frames
from duplexomni.gateway.floor import Floor, FloorController
from duplexomni.gateway.protocol import ProtocolError, decode_audio, encode_audio
from duplexomni.gateway.router import route_utterance
from duplexomni.gateway.tools import ToolError, safe_calc
from duplexomni.gateway.vad import rms


def test_barge_in_cuts_speech_and_cancels_task():
    floor = FloorController()
    floor.begin_speech("我先说完", 2.0, now=0.0)
    floor.start_thinking("t1")
    cmds = floor.preempt("barge-in")
    names = [c.name for c in cmds]
    assert names == ["cut", "floor", "cancel_task"]
    assert floor.floor == Floor.IDLE
    assert floor.task_id is None
    assert floor.user_voice(now=0.1)[0].data["floor"] == "user"


def test_speech_defers_while_user_holds_floor():
    floor = FloorController()
    floor.user_voice(now=0.0)
    cmds = floor.begin_speech("稍后", 1.0, now=0.2)
    assert cmds[0].name == "defer_speech"
    assert floor.floor == Floor.USER
    floor.user_release(now=1.0)
    spoken = floor.begin_speech("稍后", 1.0, now=1.0)
    assert spoken[0].name == "speak"
    assert floor.tick(now=3.0)[0].data["floor"] == "idle"


def test_route_and_calc():
    assert route_utterance("你好") == "chat"
    assert route_utterance("帮我算 12*(3+4)") == "delegate"
    assert route_utterance("现在几点") == "delegate"
    assert safe_calc("12*(3+4)") == "84"
    with pytest.raises(ToolError):
        safe_calc("__import__('os')")
    with pytest.raises(ToolError):
        safe_calc("2**99")


def test_client_playback_holds_the_floor_past_the_estimate():
    floor = FloorController()
    floor.begin_speech("你好", 0.4, now=0.0)
    assert floor.tick(now=0.5)[0].data["floor"] == "idle"

    live = FloorController()
    live.begin_speech("你好", 0.4, now=0.0)
    assert live.note_playback(True, now=0.2) == []
    assert live.tick(now=2.0) == []
    assert live.floor == Floor.ASSISTANT
    live.note_playback(False, now=2.0)
    assert live.tick(now=2.2)[0].data["floor"] == "idle"


def test_sent_pcm_extends_the_floor_hold():
    floor = FloorController()
    floor.begin_speech("你好", 0.05, now=0.0)
    floor.note_sent_frame(now=0.0, frame_s=0.2)
    floor.note_sent_frame(now=0.0, frame_s=0.2)
    assert floor.server_playing(0.3)
    assert floor.tick(now=0.3) == []
    assert floor.tick(now=0.41)[0].data["floor"] == "idle"


def test_playback_end_does_not_clip_audio_already_sent():
    floor = FloorController()
    floor.begin_speech("你好", 0.4, now=0.0)
    floor.note_sent_frame(now=0.1, frame_s=1.0)
    floor.note_playback(True, now=0.2)
    floor.note_playback(False, now=0.3)
    assert floor.tick(now=0.5) == []
    assert floor.server_playing(0.5)
    assert floor.tick(now=1.2)[0].data["floor"] == "idle"


def test_scripted_tone_is_audible_and_marked():
    frame = scripted_frames("好")[0]
    assert decode_scripted(frame) == (0, ord("好"))
    assert rms(frame) > 0.08


def test_audio_frame_roundtrip():
    pcm = b"\x01\x00" * 32
    raw = encode_audio(7, pcm, speech_hint=True, playback=True)
    index, body, hint, playback = decode_audio(raw)
    assert (index, body, hint, playback) == (7, pcm, True, True)
    with pytest.raises(ProtocolError):
        decode_audio(b"nope")
