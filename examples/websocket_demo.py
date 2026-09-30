"""Full-duplex websocket demo (minimal RFC 6455, stdlib only).

Server (runs the interaction layer on 480 ms slices):

    python examples/websocket_demo.py --serve --port 8765

Client (simulates a mic: silence + scripted speech, plays back assistant
audio, prints events):

    python examples/websocket_demo.py --client --port 8765 --slices 24

Protocol (JSON text frames):
    client -> server: {"type": "slice", "pcm_b64": "...", "text": "..."}
    server -> client: {"type": "slice_out", "index": i, "text": "...",
                       "pcm_b64": "...", "events": [...], "rtf": 0.1}
    client -> server: {"type": "bye"}  ends the session
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import argparse
import asyncio
import base64
import hashlib
import json
import struct

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# --------------------------------------------------------------------------- #
# Minimal RFC 6455 framing
# --------------------------------------------------------------------------- #


def encode_frame(payload: bytes, opcode: int = 1) -> bytes:
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([n])
    elif n < 65536:
        header += bytes([126]) + struct.pack(">H", n)
    else:
        header += bytes([127]) + struct.pack(">Q", n)
    return header + payload


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes] | None:
    head = await reader.readexactly(2)
    fin_op = head[0]
    opcode = fin_op & 0x0F
    masked = bool(head[1] & 0x80)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack(">H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if masked else None
    payload = await reader.readexactly(length) if length else b""
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, payload


async def handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                    *, is_server: bool) -> None:
    if is_server:
        raw = await reader.readuntil(b"\r\n\r\n")
        headers = {}
        for line in raw.decode("latin-1").split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        key = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        writer.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )
        await writer.drain()
    else:
        import os

        key = base64.b64encode(os.urandom(16)).decode()
        writer.write(
            (
                f"GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        await writer.drain()
        await reader.readuntil(b"\r\n\r\n")


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


async def serve(port: int) -> None:
    import torch

    from duplexomni.config import tiny_config
    from duplexomni.model import DuplexOmni
    from duplexomni.runtime import InteractionLayer, ThinkingBridge, UserSliceInput
    from duplexomni.runtime.thinking import EchoThinkingLayer

    torch.manual_seed(0)
    model = DuplexOmni(tiny_config())
    print(f"model ready ({sum(p.numel() for p in model.parameters()):,} params); listening on {port}")

    async def client_handler(reader, writer):
        peer = writer.get_extra_info("peername")
        print("client connected:", peer)
        await handshake(reader, writer, is_server=True)
        queue: asyncio.Queue = asyncio.Queue()

        async def reader_loop():
            try:
                while True:
                    frame = await read_frame(reader)
                    if frame is None:
                        break
                    opcode, payload = frame
                    if opcode == 8:  # close
                        await queue.put(None)
                        break
                    if opcode == 9:  # ping -> pong
                        writer.write(encode_frame(payload, opcode=10))
                        await writer.drain()
                        continue
                    if opcode != 1:
                        continue
                    msg = json.loads(payload.decode("utf-8"))
                    if msg.get("type") == "bye":
                        await queue.put(None)
                        break
                    await queue.put(
                        UserSliceInput(
                            index=msg.get("index", 0),
                            pcm=base64.b64decode(msg.get("pcm_b64", "")),
                            asr_text=msg.get("text"),
                        )
                    )
            except (asyncio.IncompleteReadError, ConnectionResetError):
                await queue.put(None)

        async def user_stream():
            index = 0
            while True:
                item = await queue.get()
                if item is None:
                    break
                item.index = index
                yield item
                index += 1

        async def run_session():
            bridge = ThinkingBridge(EchoThinkingLayer(delay_s=0.0, n_fragments=3))
            layer = InteractionLayer(model, bridge=bridge, tokens_per_slice=6)
            async for out in layer.run(user_stream()):
                msg = {
                    "type": "slice_out",
                    "index": out.index,
                    "text": out.text,
                    "pcm_b64": base64.b64encode(out.pcm).decode("ascii"),
                    "events": [type(e).__name__ for e in out.events],
                    "rtf": round(out.rtf, 4),
                    "user_speaking": out.user_speaking,
                    "assistant_speaking": out.assistant_speaking,
                }
                writer.write(encode_frame(json.dumps(msg).encode("utf-8")))
                await writer.drain()
            writer.write(encode_frame(b"", opcode=8))
            await writer.drain()
            writer.close()

        readers = asyncio.gather(reader_loop(), run_session(), return_exceptions=True)
        await readers
        print("client disconnected:", peer)

    server = await asyncio.start_server(client_handler, "0.0.0.0", port)
    async with server:
        await server.serve_forever()


# --------------------------------------------------------------------------- #
# Client (simulated microphone)
# --------------------------------------------------------------------------- #


async def client(host: str, port: int, slices: int, wav_out: str | None) -> None:
    from duplexomni.data import MockTTS

    reader, writer = await asyncio.open_connection(host, port)
    await handshake(reader, writer, is_server=False)

    tts = MockTTS()
    speech = tts.synthesize("hello there, I have a quick question for you", speaker="user").pcm
    speech2 = tts.synthesize("wait, actually let me change the plan", speaker="user").pcm
    speech_at = {3: (speech, "hello there, I have a quick question for you"),
                 12: (speech2, "wait, actually let me change the plan")}

    import os

    def masked_frame(payload: bytes) -> bytes:
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        n = len(payload)
        if n < 126:
            header = bytes([0x81, 0x80 | n])
        elif n < 65536:
            header = bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
        else:
            header = bytes([0x81, 0x80 | 127]) + struct.pack(">Q", n)
        return header + mask + masked

    async def sender():
        for i in range(slices):
            if i in speech_at:
                pcm, text = speech_at[i]
                msg = {"type": "slice", "pcm_b64": base64.b64encode(pcm[:15360]).decode(),
                       "text": text}
            else:
                msg = {"type": "slice", "pcm_b64": base64.b64encode(b"\x00" * 15360).decode()}
            writer.write(masked_frame(json.dumps(msg).encode()))
            await writer.drain()
            await asyncio.sleep(0.48)
        writer.write(masked_frame(json.dumps({"type": "bye"}).encode()))
        await writer.drain()

    audio = bytearray()

    async def receiver():
        while True:
            frame = await read_frame(reader)
            if frame is None or frame[0] == 8:
                break
            if frame[0] != 1:
                continue
            msg = json.loads(frame[1].decode())
            if msg.get("type") != "slice_out":
                continue
            audio.extend(base64.b64decode(msg["pcm_b64"]))
            evs = ",".join(msg["events"]) or "-"
            print(f"[{msg['index']:3d}] rtf={msg['rtf']:.2f} "
                  f"u={'Y' if msg['user_speaking'] else '-'} "
                  f"a={'Y' if msg['assistant_speaking'] else '-'} events={evs}")

    await asyncio.gather(sender(), receiver())
    writer.close()
    print(f"received {len(audio)} bytes of assistant audio "
          f"({len(audio) / 32000:.1f}s @16k mono)")
    if wav_out and audio:
        from duplexomni.codecs import write_wav

        write_wav(wav_out, bytes(audio))
        print(f"wrote assistant audio -> {wav_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--client", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--slices", type=int, default=24)
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()

    if args.serve:
        asyncio.run(serve(args.port))
    elif args.client:
        asyncio.run(client(args.host, args.port, args.slices, args.wav))
    else:
        ap.error("choose --serve or --client")


if __name__ == "__main__":
    main()
