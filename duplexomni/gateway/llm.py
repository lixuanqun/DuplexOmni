"""Minimal OpenAI-compatible chat client with cancellable streaming.

Stdlib only: ``asyncio.open_connection`` plus TLS. Closing the writer
cancels a read, which is what barge-in needs. The client does not follow
redirects and caps the response body.
"""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator
from urllib.parse import urlparse

_MAX_BODY = 1_000_000


class LLMError(RuntimeError):
    """The thinking-layer endpoint failed or spoke a protocol we reject."""


class OpenAIChat:
    def __init__(self, base_url: str, model: str, api_key: str = "", *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def short_reply(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是全双工语音交互层。只用一两句口语回答，不要调用工具，不要列清单。",
            },
            {"role": "user", "content": text},
        ]
        chunks: list[str] = []
        async for delta in self.stream_chat(messages, max_tokens=80, temperature=0.3):
            chunks.append(delta)
            if sum(len(part) for part in chunks) > 200:
                break
        return "".join(chunks).strip()

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async for data in post_sse(
            f"{self.base_url}/chat/completions",
            payload,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
        ):
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                yield content


async def post_sse(
    url: str,
    payload: dict,
    *,
    api_key: str = "",
    timeout_s: float = 30.0,
) -> AsyncIterator[str]:
    """Yield SSE ``data:`` payloads from a POST."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMError("thinking endpoint must be http or https")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    body = json.dumps(payload).encode("utf-8")
    header_lines = [
        f"POST {path} HTTP/1.1",
        f"Host: {parsed.hostname}" + (f":{port}" if parsed.port else ""),
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Accept: text/event-stream",
        "Connection: close",
    ]
    if api_key:
        header_lines.append(f"Authorization: Bearer {api_key}")
    raw_request = ("\r\n".join(header_lines) + "\r\n\r\n").encode("ascii") + body
    ssl_ctx = ssl.create_default_context() if parsed.scheme == "https" else None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                parsed.hostname,
                port,
                ssl=ssl_ctx,
                server_hostname=parsed.hostname if ssl_ctx else None,
            ),
            timeout=min(5.0, timeout_s),
        )
    except Exception as exc:
        raise LLMError("thinking endpoint unreachable") from exc

    try:
        writer.write(raw_request)
        await writer.drain()
        status, header_map, leftover = await asyncio.wait_for(_read_head(reader), timeout=timeout_s)
        if status != 200:
            err = leftover + await _read_some(reader, 4096)
            raise LLMError(f"thinking endpoint status {status}: {err[:180]!r}")
        sse = _SSEParser()
        received = 0
        async for chunk in _iter_body(reader, leftover, header_map):
            received += len(chunk)
            if received > _MAX_BODY:
                raise LLMError("thinking response too large")
            for item in sse.feed(chunk.decode("utf-8", errors="replace")):
                yield item
        for item in sse.feed("\n", flush=True):
            yield item
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1)
        except Exception:
            pass


class _SSEParser:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str, *, flush: bool = False) -> list[str]:
        self._buf += text
        out: list[str] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            item = _sse_data(line)
            if item is not None:
                out.append(item)
        if flush and self._buf.strip():
            item = _sse_data(self._buf)
            self._buf = ""
            if item is not None:
                out.append(item)
        return out


def _sse_data(line: str) -> str | None:
    stripped = line.strip("\r").strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[5:].strip()


async def _read_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str], bytes]:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await reader.read(1024)
        if not chunk:
            raise LLMError("thinking endpoint closed before headers")
        data += chunk
        if len(data) > 65_536:
            raise LLMError("thinking response headers too large")
    head, leftover = data.split(b"\r\n\r\n", 1)
    lines = head.decode("latin-1", errors="replace").split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        raise LLMError("malformed thinking response")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return int(parts[1]), headers, leftover


async def _read_some(reader: asyncio.StreamReader, limit: int) -> bytes:
    try:
        return await asyncio.wait_for(reader.read(limit), timeout=2)
    except Exception:
        return b""


async def _iter_body(
    reader: asyncio.StreamReader,
    leftover: bytes,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    encoding = headers.get("transfer-encoding", "")
    if "chunked" in encoding:
        buf = leftover
        while True:
            while b"\r\n" not in buf:
                more = await reader.read(1024)
                if not more:
                    return
                buf += more
            line, buf = buf.split(b"\r\n", 1)
            size_text = line.split(b";", 1)[0].strip()
            if not size_text:
                continue
            try:
                size = int(size_text, 16)
            except ValueError as exc:
                raise LLMError("bad chunk size") from exc
            if size == 0:
                return
            while len(buf) < size + 2:
                more = await reader.read(size + 2 - len(buf))
                if not more:
                    return
                buf += more
            yield buf[:size]
            buf = buf[size + 2 :]
    length = int(headers.get("content-length", "0") or "0")
    if length:
        buf = leftover
        while len(buf) < length:
            more = await reader.read(min(4096, length - len(buf)))
            if not more:
                break
            buf += more
        if buf:
            yield buf[:length]
        return
    if leftover:
        yield leftover
    while True:
        more = await reader.read(4096)
        if not more:
            return
        yield more
