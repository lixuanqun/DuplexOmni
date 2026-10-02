"""Binary audio frames and JSON control messages.

Audio is a 12-byte header plus 16-bit little-endian PCM. Control messages
stay JSON so a session can be debugged with a generic WebSocket client.
The cognitive slice is still 480 ms; the wire frame is 20 ms so barge-in
does not wait for the next slice.
"""

from __future__ import annotations

import json
import struct
from typing import Any

MAGIC = b"PCM1"
VERSION = 1
HEADER_LEN = 12

FLAG_SPEECH = 0x01
FLAG_PLAYBACK = 0x02

JSON_TYPES = frozenset({
    "hello", "text", "transcript", "barge", "cancel", "ping", "bye", "playback", "auth",
})


class ProtocolError(ValueError):
    """A frame that violates the duplex wire contract."""


def encode_audio(index: int, pcm: bytes, *, speech_hint: bool = False, playback: bool = False) -> bytes:
    if index < 0 or index > 0xFFFFFFFF:
        raise ProtocolError("audio index out of range")
    flags = (FLAG_SPEECH if speech_hint else 0) | (FLAG_PLAYBACK if playback else 0)
    return MAGIC + bytes((VERSION, flags, 0, 0)) + struct.pack("<I", index) + pcm


def decode_audio(buf: bytes, *, max_pcm: int = 32_000) -> tuple[int, bytes, bool, bool]:
    if len(buf) < HEADER_LEN or buf[:4] != MAGIC:
        raise ProtocolError("audio frame missing PCM1 header")
    if buf[4] != VERSION:
        raise ProtocolError(f"unsupported audio version {buf[4]}")
    if len(buf) - HEADER_LEN > max_pcm:
        raise ProtocolError("audio frame too large")
    if (len(buf) - HEADER_LEN) % 2:
        raise ProtocolError("pcm length must be even")
    flags = buf[5]
    index = struct.unpack("<I", buf[8:12])[0]
    return index, buf[HEADER_LEN:], bool(flags & FLAG_SPEECH), bool(flags & FLAG_PLAYBACK)


def encode_json(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def decode_json(raw: str, *, max_text: int = 4_000) -> dict[str, Any]:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("control frame is not json") from exc
    if not isinstance(message, dict):
        raise ProtocolError("control frame must be an object")
    kind = message.get("type")
    if kind not in JSON_TYPES:
        raise ProtocolError("unknown control type")
    text = message.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise ProtocolError("text must be a string")
        if len(text) > max_text:
            message = dict(message)
            message["text"] = text[:max_text]
    return message
