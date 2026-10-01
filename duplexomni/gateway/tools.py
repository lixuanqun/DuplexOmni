"""Safe tools the thinking worker is allowed to run.

There is no shell and no open network. ``calculate`` evaluates a restricted
expression tree. Memory stays inside the session. Add tools by registering
them; do not widen the defaults into arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_EXPR_CHARS = re.compile(r"[0-9.+\-*/%()\s×÷（）]+")


class ToolError(Exception):
    """A tool refused its input or failed in a way the user can hear."""


@dataclass
class Memory:
    notes: list[str] = field(default_factory=list)
    limit: int = 50

    def add(self, text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        self.notes.append(cleaned[:500])
        del self.notes[:-self.limit]

    def render(self) -> str:
        if not self.notes:
            return ""
        return "\n".join(f"- {note}" for note in self.notes[-10:])


@dataclass
class ToolContext:
    goal: str
    memory: Memory


ToolFn = Callable[[dict[str, Any], ToolContext], str | Awaitable[str]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolFn, float]] = {}

    def register(self, name: str, fn: ToolFn, *, timeout_s: float = 4.0) -> None:
        self._tools[name] = (fn, timeout_s)

    def timeout_of(self, name: str) -> float:
        if name not in self._tools:
            return 1.0
        return self._tools[name][1]

    def known(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str:
        if name not in self._tools:
            raise ToolError(f"没有这个工具：{name}")
        fn = self._tools[name][0]
        result = fn(args, ctx)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        return str(result)


def safe_calc(expression: str) -> str:
    compact = (
        expression.replace("×", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
        .strip()
    )
    if not compact or len(compact) > 64:
        raise ToolError("式子为空或太长")
    if not re.fullmatch(r"[0-9.+\-*/%()\s]+", compact):
        raise ToolError("式子里有不能计算的字符")
    try:
        tree = ast.parse(compact, mode="eval")
        value = _eval_node(tree.body)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError("这个式子我没法安全计算") from exc
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 8 or abs(left) > 1_000_000):
            raise ToolError("指数太大")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("不能除以零")
        return _OPS[type(node.op)](left, right)
    raise ToolError("这个式子我没法安全计算")


def extract_expression(text: str) -> str | None:
    translated = text.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")
    match = _EXPR_CHARS.search(translated)
    if not match:
        return None
    expr = "".join(match.group(0).split())
    if not any(ch.isdigit() for ch in expr) or not any(op in expr for op in "+-*/%"):
        return None
    return expr


def _tool_now(args: dict[str, Any], ctx: ToolContext) -> str:
    del args, ctx
    return datetime.now().strftime("现在是 %Y-%m-%d %H:%M:%S")


def _tool_calculate(args: dict[str, Any], ctx: ToolContext) -> str:
    expression = str(args.get("expression") or extract_expression(ctx.goal) or "")
    value = safe_calc(expression)
    return f"{expression} = {value}"


def _tool_remember(args: dict[str, Any], ctx: ToolContext) -> str:
    text = str(args.get("text") or "").strip()
    if not text:
        raise ToolError("没有要记住的内容")
    ctx.memory.add(text)
    return f"我记住了：{text}"


def _tool_recall(args: dict[str, Any], ctx: ToolContext) -> str:
    del args
    rendered = ctx.memory.render()
    if not rendered:
        return "我还没有记住任何内容。"
    return "我记得这些：\n" + rendered


def _tool_reason(args: dict[str, Any], ctx: ToolContext) -> str:
    goal = str(args.get("goal") or ctx.goal).strip()
    notes = ctx.memory.render() or "没有记下的内容"
    return f"我在本地看过这件事：「{goal[:80]}」。已记住的内容：{notes}。"


def rule_plan(goal: str) -> list[tuple[str, dict[str, Any]]]:
    """Map an utterance onto an ordered tool plan. Empty means a local reason step."""
    steps: list[tuple[str, dict[str, Any]]] = []
    remember = re.search(r"(?:记住|帮我记)[:：\s]*(.+)", goal)
    if remember:
        steps.append(("remember", {"text": remember.group(1).strip()}))
    expression = extract_expression(goal)
    if expression:
        steps.append(("calculate", {"expression": expression}))
    if any(phrase in goal for phrase in ("几点", "现在时间", "什么时间", "今天几号", "日期")):
        steps.append(("now", {}))
    if any(phrase in goal for phrase in ("回忆", "我让你记", "你记了")):
        steps.append(("recall", {}))
    if not steps:
        steps.append(("reason", {"goal": goal}))
    return steps


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("now", _tool_now, timeout_s=1.0)
    registry.register("calculate", _tool_calculate, timeout_s=1.0)
    registry.register("remember", _tool_remember, timeout_s=1.0)
    registry.register("recall", _tool_recall, timeout_s=1.0)
    registry.register("reason", _tool_reason, timeout_s=1.0)
    return registry
