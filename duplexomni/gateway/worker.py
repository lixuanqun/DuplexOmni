"""Thinking worker: plan, run tools, stream fragments.

The worker never touches the socket. It reports events; the session decides
whether they may be spoken. Cancellation is cooperative between steps and
preemptive via task cancellation inside a tool wait.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .tools import ToolContext, ToolError, ToolRegistry, rule_plan

EventFn = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class TaskSpec:
    task_id: str
    goal: str
    session_id: str = ""


async def execute_task(
    spec: TaskSpec,
    *,
    tools: ToolRegistry,
    memory: Any,
    on_event: EventFn,
    cancel: asyncio.Event,
    planner: Callable[[str], list[tuple[str, dict[str, Any]]]] = rule_plan,
    task_timeout_s: float = 30.0,
    llm: Any | None = None,
) -> None:
    started = time.monotonic()
    await on_event({"kind": "status", "status": "running", "goal": spec.goal})
    ctx = ToolContext(goal=spec.goal, memory=memory)
    try:
        if llm is not None:
            try:
                await _execute_llm(
                    spec, tools=tools, ctx=ctx, on_event=on_event, cancel=cancel,
                    llm=llm, deadline=started + task_timeout_s,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                if cancel.is_set() or time.monotonic() >= started + task_timeout_s:
                    raise
                await on_event({
                    "kind": "fragment",
                    "text": "思考层这次没有接上，我改在本地把能做的做完。",
                })
        await _execute_rules(
            spec, tools=tools, ctx=ctx, on_event=on_event, cancel=cancel,
            planner=planner, deadline=started + task_timeout_s,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await on_event({"kind": "status", "status": "failed", "detail": str(exc)[:200]})


async def _execute_rules(
    spec: TaskSpec,
    *,
    tools: ToolRegistry,
    ctx: ToolContext,
    on_event: EventFn,
    cancel: asyncio.Event,
    planner: Callable[[str], list[tuple[str, dict[str, Any]]]],
    deadline: float,
) -> None:
    steps = planner(spec.goal)
    pieces: list[str] = []
    for name, args in steps:
        if cancel.is_set() or time.monotonic() >= deadline:
            return
        await on_event({"kind": "status", "status": "tool", "tool": name, "args": args})
        try:
            result = await asyncio.wait_for(
                tools.call(name, args, ctx),
                timeout=min(tools.timeout_of(name), max(0.1, deadline - time.monotonic())),
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            result = f"{name} 超时了。"
        except ToolError as exc:
            result = str(exc)
        except Exception:
            result = f"{name} 执行失败。"
        if cancel.is_set():
            return
        pieces.append(result)
        await on_event({"kind": "fragment", "text": result})
    if cancel.is_set():
        return
    summary = "；".join(pieces)[:500]
    await on_event({"kind": "status", "status": "done", "summary": summary})


async def _execute_llm(
    spec: TaskSpec,
    *,
    tools: ToolRegistry,
    ctx: ToolContext,
    on_event: EventFn,
    cancel: asyncio.Event,
    llm: Any,
    deadline: float,
) -> None:
    """Up to four think/tool rounds. Lines are SAY / TOOL / DONE."""
    system = (
        "你是全双工语音系统的思考层。交互层已经在和用户说话，你负责把任务做完。"
        "只用一行一条的纯文本，不要 Markdown。"
        "命令：SAY 短句；TOOL calculate {\"expression\":\"1+2\"}；"
        "TOOL now {}；TOOL remember {\"text\":\"...\"}；TOOL recall {}；"
        "DONE 一句话结果。先说进度，再调工具，最后 DONE。不要编造工具结果。"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": spec.goal},
    ]
    spoken: list[str] = []
    for _round in range(4):
        if cancel.is_set() or time.monotonic() >= deadline:
            return
        buffer = ""
        full: list[str] = []
        tool_calls: list[tuple[str, dict]] = []
        async for delta in llm.stream_chat(messages):
            if cancel.is_set() or time.monotonic() >= deadline:
                return
            full.append(delta)
            buffer += delta
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                await _handle_line(line, tools, ctx, on_event, spoken, tool_calls, cancel)
        if buffer.strip():
            await _handle_line(buffer, tools, ctx, on_event, spoken, tool_calls, cancel)
        if not tool_calls:
            break
        assistant_text = "".join(full)
        results = []
        for name, args in tool_calls:
            if cancel.is_set():
                return
            await on_event({"kind": "status", "status": "tool", "tool": name, "args": args})
            try:
                result = await asyncio.wait_for(tools.call(name, args, ctx), timeout=tools.timeout_of(name))
            except Exception as exc:
                result = str(exc)
            results.append(f"{name}: {result}")
            await on_event({"kind": "fragment", "text": result})
            spoken.append(result)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": "工具结果：\n" + "\n".join(results)})
        if "DONE" in assistant_text.upper():
            break
    summary = "；".join(spoken)[:500] or "思考层没有给出可说的结果。"
    await on_event({"kind": "status", "status": "done", "summary": summary})


async def _handle_line(
    line: str,
    tools: ToolRegistry,
    ctx: ToolContext,
    on_event: EventFn,
    spoken: list[str],
    tool_calls: list[tuple[str, dict]],
    cancel: asyncio.Event,
) -> None:
    del ctx
    if cancel.is_set():
        return
    text = line.strip()
    if not text:
        return
    say = re.match(r"SAY\s+(.*)", text, flags=re.IGNORECASE)
    done = re.match(r"DONE\s*(.*)", text, flags=re.IGNORECASE)
    tool = re.match(r"TOOL\s+([A-Za-z_]\w*)\s*(\{.*\})?\s*$", text, flags=re.IGNORECASE)
    if say:
        uttered = say.group(1).strip()
        if uttered:
            spoken.append(uttered)
            await on_event({"kind": "fragment", "text": uttered})
        return
    if done:
        uttered = done.group(1).strip()
        if uttered:
            spoken.append(uttered)
            await on_event({"kind": "fragment", "text": uttered})
        return
    if tool:
        name = tool.group(1)
        raw_args = tool.group(2) or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if tools.known(name):
            tool_calls.append((name, args))
        else:
            await on_event({"kind": "fragment", "text": f"不能使用工具 {name}。"})
        return
    spoken.append(text[:200])
    await on_event({"kind": "fragment", "text": text[:200]})
