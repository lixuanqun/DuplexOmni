"""Streaming chat client against a local HTTP stub."""

from __future__ import annotations

import asyncio

from duplexomni.gateway.llm import post_sse


def test_sse_content_deltas():
    async def run() -> None:
        payload = (
            b'data: {"choices":[{"delta":{"content":"SAY "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"ok\\n"}}]}\n\n'
            b"data: [DONE]\n\n"
        )

        async def handler(reader, writer):
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = await reader.read(1024)
                if not chunk:
                    writer.close()
                    return
                data += chunk
            head, rest = data.split(b"\r\n\r\n", 1)
            length = 0
            for line in head.decode("latin-1").split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1])
            buf = rest
            while len(buf) < length:
                buf += await reader.read(length - len(buf))
            body = (
                "HTTP/1.1 200 OK\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Content-Type: text/event-stream\r\n"
                "Connection: close\r\n\r\n"
            ).encode() + payload
            writer.write(body)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            parts = [
                item
                async for item in post_sse(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"model": "stub"},
                    timeout_s=3,
                )
            ]
        finally:
            server.close()
            await server.wait_closed()
        assert parts == [
            '{"choices":[{"delta":{"content":"SAY "}}]}',
            '{"choices":[{"delta":{"content":"ok\\n"}}]}',
            "[DONE]",
        ]

    asyncio.run(run())
