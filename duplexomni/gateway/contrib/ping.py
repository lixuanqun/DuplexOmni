"""Example contrib tool. It does not touch the network or the shell."""

from __future__ import annotations

from ..tools import ToolRegistry


def register(registry: ToolRegistry) -> None:
    registry.register("ping", _ping, timeout_s=1.0)


def _ping(args: dict, ctx: object) -> str:
    del args, ctx
    return "pong"
