"""WebSocket gateway and the static full-duplex client.

One process serves ``/`` (the browser) and ``/ws`` (the session). Binding
defaults to loopback. Each connection is one session; overload closes the
socket instead of queueing work without a bound.
"""

from __future__ import annotations

import asyncio
import http
import json
import logging
import os
import re
from email.utils import formatdate
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from .config import GatewayConfig
from .llm import OpenAIChat
from .metrics import METERS
from .protocol import ProtocolError, decode_audio, decode_json, encode_json
from .session import DuplexSession
from .store import SessionStore

log = logging.getLogger("duplexomni.gateway")

_WEB_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/audio-capture.worklet.js": ("audio-capture.worklet.js", "text/javascript; charset=utf-8"),
    "/audio-playback.worklet.js": ("audio-playback.worklet.js", "text/javascript; charset=utf-8"),
}

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{4,64}")


class ConnectionGate:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.total = 0
        self.per_ip: dict[str, int] = {}

    def acquire(self, ip: str) -> str | None:
        if self.total >= self.config.max_sessions:
            return "overloaded"
        if self.per_ip.get(ip, 0) >= self.config.max_sessions_per_ip:
            return "too many sessions from this address"
        self.total += 1
        self.per_ip[ip] = self.per_ip.get(ip, 0) + 1
        METERS.set_gauge("sessions", self.total)
        return None

    def release(self, ip: str) -> None:
        self.total = max(0, self.total - 1)
        left = self.per_ip.get(ip, 0) - 1
        if left <= 0:
            self.per_ip.pop(ip, None)
        else:
            self.per_ip[ip] = left
        METERS.set_gauge("sessions", self.total)


def web_root() -> Path:
    env = os.environ.get("DUPLEX_WEB_ROOT")
    if env:
        return Path(env)
    packaged = Path(__file__).resolve().parent / "webassets"
    if (packaged / "index.html").exists():
        return packaged
    return Path(__file__).resolve().parents[2] / "web"


def _response(status: int, body: bytes, content_type: str) -> Response:
    phrase = http.HTTPStatus(status).phrase
    headers = Headers([
        ("Date", formatdate(usegmt=True)),
        ("Connection", "close"),
        ("Content-Length", str(len(body))),
        ("Content-Type", content_type),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
    ])
    return Response(status, phrase, headers, body)


def credentials_ok(config: GatewayConfig, peer: str, supplied: str) -> bool:
    """A configured token must match. With no token, only loopback peers connect."""
    if config.token:
        return supplied == config.token
    return peer in _LOOPBACK


def _supplied_token(request) -> str:
    header = request.headers.get("Authorization") or request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        bearer = header[7:].strip()
        if bearer:
            return bearer
    query = parse_qs(urlsplit(request.path).query)
    values = query.get("token") or []
    return values[0] if values else ""


def _peer(connection) -> str:
    address = getattr(connection, "remote_address", None)
    if not address:
        return ""
    return str(address[0])


def _session_id_from(request) -> str | None:
    query = parse_qs(urlsplit(getattr(request, "path", "")).query)
    values = query.get("session_id") or []
    if not values:
        return None
    candidate = values[0]
    if _SESSION_ID.fullmatch(candidate):
        return candidate
    return None


def _http(request) -> Response | None:
    path = request.path.split("?", 1)[0]
    if path == "/ws":
        return None
    if path == "/health":
        body = json.dumps({"ok": True, "sessions": METERS.counters.get("sessions", 0)}).encode()
        return _response(200, body, "application/json")
    if path == "/metrics":
        return _response(200, METERS.render().encode(), "text/plain; version=0.0.4; charset=utf-8")
    spec = _WEB_FILES.get(path)
    if spec is None:
        return _response(404, b"not found", "text/plain; charset=utf-8")
    file_path = web_root() / spec[0]
    if not file_path.is_file():
        return _response(404, b"ui not built", "text/plain; charset=utf-8")
    return _response(200, file_path.read_bytes(), spec[1])


async def _send_loop(ws, session: DuplexSession) -> None:
    json_get = asyncio.create_task(session.outbound.get())
    pcm_get = asyncio.create_task(session.pcm_out.get())
    try:
        while True:
            done, _pending = await asyncio.wait(
                {json_get, pcm_get},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if json_get in done:
                message = json_get.result()
                json_get = asyncio.create_task(session.outbound.get())
                await ws.send(encode_json(message))
            if pcm_get in done:
                item = pcm_get.result()
                pcm_get = asyncio.create_task(session.pcm_out.get())
                frame = session.live_pcm(item)
                if frame is not None:
                    await ws.send(frame)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.info("session %s sender stopped", session.session_id)
    finally:
        json_get.cancel()
        pcm_get.cancel()
        await asyncio.gather(json_get, pcm_get, return_exceptions=True)


async def _handle(
    ws,
    config: GatewayConfig,
    gate: ConnectionGate,
    llm: OpenAIChat | None,
    store: SessionStore,
) -> None:
    peer = "unknown"
    if ws.remote_address:
        peer = str(ws.remote_address[0])
    reason = gate.acquire(peer)
    if reason:
        await ws.close(1013, reason)
        METERS.inc("rejected_sessions")
        return
    request = getattr(ws, "request", None)
    session_id = _session_id_from(request) if request is not None else None
    session = DuplexSession(config, llm=llm, session_id=session_id, store=store)
    store.ensure_session(session.session_id)
    session.start()
    session.emit(session.hello_message())
    session.replay()
    sender = asyncio.create_task(_send_loop(ws, session), name=f"duplex-send-{session.session_id}")
    log.info("session %s open peer=%s", session.session_id, peer)
    try:
        async for incoming in ws:
            if isinstance(incoming, bytes):
                try:
                    index, pcm, hint, playback = decode_audio(incoming, max_pcm=config.max_pcm_bytes)
                except ProtocolError:
                    session._bad_messages += 1
                    session.emit({"type": "error", "code": "protocol", "message": "音频帧无效"})
                else:
                    session.ingest_audio(index, pcm, hint, playback)
            else:
                try:
                    message = decode_json(incoming, max_text=config.max_text)
                except ProtocolError:
                    session._bad_messages += 1
                    session.emit({"type": "error", "code": "protocol", "message": "控制帧无效"})
                else:
                    if message.get("type") == "bye":
                        break
                    await session.ingest_json(message)
            if session._bad_messages >= config.max_bad_messages:
                session.emit({"type": "error", "code": "protocol", "message": "无效帧过多，关闭会话"})
                break
    except Exception:
        log.exception("session %s crashed", session.session_id)
        METERS.inc("session_crashes")
    finally:
        sender.cancel()
        await session.shutdown()
        gate.release(peer)
        log.info("session %s closed", session.session_id)
        try:
            await sender
        except asyncio.CancelledError:
            pass


def build_llm(config: GatewayConfig) -> OpenAIChat | None:
    if not config.llm_enabled:
        return None
    return OpenAIChat(config.llm_base_url, config.llm_model, config.llm_api_key, timeout_s=config.task_timeout_s)


def open_gateway(config: GatewayConfig):
    """Return the websockets server context manager. ``port=0`` picks a free port."""
    gate = ConnectionGate(config)
    llm = build_llm(config)
    store = SessionStore(config.db_path)

    def process_request(connection, request):
        path = request.path.split("?", 1)[0]
        if path == "/ws" and not credentials_ok(config, _peer(connection), _supplied_token(request)):
            METERS.inc("rejected_sessions")
            return _response(401, b"unauthorized", "text/plain; charset=utf-8")
        if path == "/ws":
            return None
        return _http(request)

    return serve(
        lambda ws: _handle(ws, config, gate, llm, store),
        config.host,
        config.port,
        process_request=process_request,
        max_size=65_536,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=10,
    )


async def serve_gateway(config: GatewayConfig) -> None:
    root = web_root()
    log.info("ui root %s", root)
    async with open_gateway(config):
        log.info(
            "duplex gateway http://%s:%s  ws://%s:%s/ws",
            config.host, config.port, config.host, config.port,
        )
        await asyncio.Future()


def serve_forever(config: GatewayConfig | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    cfg = config or GatewayConfig.from_env()
    try:
        asyncio.run(serve_gateway(cfg))
    except KeyboardInterrupt:
        log.info("gateway stopped")
    return 0
