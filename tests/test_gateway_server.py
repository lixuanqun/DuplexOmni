"""Gateway socket: health, delegation, and the session cap."""

from __future__ import annotations

import asyncio
import json
import urllib.request

import pytest

websockets = pytest.importorskip("websockets")

from duplexomni.gateway.config import GatewayConfig  # noqa: E402
from duplexomni.gateway.server import credentials_ok, open_gateway  # noqa: E402


def test_websocket_delegates_and_serves_health():
    async def run() -> None:
        config = GatewayConfig(host="127.0.0.1", port=0, idle_timeout_s=30)
        async with open_gateway(config) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
                hello = json.loads(await asyncio.wait_for(ws.recv(), 2))
                assert hello["type"] == "hello"
                await ws.send(json.dumps({"type": "text", "text": "帮我算 12*(3+4)"}, ensure_ascii=False))
                seen = []
                for _ in range(80):
                    raw = await asyncio.wait_for(ws.recv(), 2)
                    if isinstance(raw, bytes):
                        continue
                    seen.append(json.loads(raw))
                    if any(item.get("status") == "done" for item in seen):
                        break
            blob = json.dumps(seen, ensure_ascii=False)
            assert "84" in blob

            def fetch() -> str:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                    return resp.read().decode()

            body = await asyncio.to_thread(fetch)
            assert '"ok": true' in body or '"ok":true' in body

    asyncio.run(run())


def test_wrong_token_is_rejected():
    async def run() -> None:
        config = GatewayConfig(host="127.0.0.1", port=0, token="secret", idle_timeout_s=30)
        async with open_gateway(config) as server:
            port = server.sockets[0].getsockname()[1]
            with pytest.raises(websockets.exceptions.InvalidStatus):
                async with websockets.connect(f"ws://127.0.0.1:{port}/ws?token=nope"):
                    pass
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/ws",
                additional_headers={"Authorization": "Bearer secret"},
            ) as ws:
                hello = json.loads(await asyncio.wait_for(ws.recv(), 2))
                assert hello["type"] == "hello"

    asyncio.run(run())


def test_credentials_allow_loopback_without_a_token():
    assert credentials_ok(GatewayConfig(), "127.0.0.1", "")
    assert not credentials_ok(GatewayConfig(), "203.0.113.8", "")
    assert credentials_ok(GatewayConfig(token="secret"), "203.0.113.8", "secret")
    assert not credentials_ok(GatewayConfig(token="secret"), "127.0.0.1", "other")


def test_second_session_is_rejected_at_capacity():
    async def run() -> None:
        config = GatewayConfig(host="127.0.0.1", port=0, max_sessions=1, idle_timeout_s=30)
        async with open_gateway(config) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as first:
                assert json.loads(await asyncio.wait_for(first.recv(), 2))["type"] == "hello"
                async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as second:
                    with pytest.raises(websockets.exceptions.ConnectionClosed):
                        await asyncio.wait_for(second.recv(), 2)

    asyncio.run(run())
