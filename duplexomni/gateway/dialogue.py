"""Short replies for the interaction layer.

These are the words spoken before a delegated task returns. They must be
immediate: the audio loop never waits on the worker to produce them.
"""

from __future__ import annotations


def estimate_hold_s(text: str) -> float:
    """How long the assistant keeps the floor while this line is spoken."""
    n = max(len(text.strip()), 1)
    return min(8.0, max(0.45, n * 0.18))


def fast_ack(text: str, route: str) -> str:
    if route == "delegate":
        return "好，我接着听，这件事交给后面做。"
    stripped = text.strip()
    if any(greet in stripped for greet in ("你好", "您好", "嗨", "hello", "hi")):
        return "你好，我在听。你可以直接说，也可以随时打断我。"
    if any(word in stripped for word in ("谢谢", "感谢")):
        return "不客气。还需要的话直接说。"
    return "我听到了。计算、看时间或者让我记住一句话，直接说就行。"
