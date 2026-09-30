"""Session state shared by both layers.

Tracks the dialogue history (spoken text + ghost text), video summary and
task state, and packages them into a :class:`ThinkingContext` when the
interaction layer dispatches a ``[THINK]`` request — the paper's "input
organisation" that its ablation shows is the source of the interaction
layer's synergy with the thinking layer (BigBench Audio 58.9 -> 77.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .thinking import ThinkingContext


@dataclass
class SessionState:
    session_id: str = "session-0"
    started_at: float = field(default_factory=time.monotonic)
    turns: list[dict] = field(default_factory=list)  # {speaker, text, ghost, t}
    task_state: dict = field(default_factory=dict)
    video_summary: str = ""
    thinking_requests: int = 0
    aborts: int = 0

    def add_user_text(self, text: str) -> None:
        if text.strip():
            self.turns.append({"speaker": "user", "text": text.strip(), "ghost": "", "t": time.monotonic()})

    def add_assistant_text(self, text: str, ghost: str = "") -> None:
        if text.strip() or ghost.strip():
            self.turns.append(
                {"speaker": "assistant", "text": text.strip(), "ghost": ghost.strip(),
                 "t": time.monotonic()}
            )

    def build_context(self) -> ThinkingContext:
        dialogue = [(t["speaker"], t["text"]) for t in self.turns[-16:]]
        return ThinkingContext(
            dialogue_text=dialogue,
            video_info=self.video_summary,
            task_state=dict(self.task_state),
        )

    @property
    def duration_s(self) -> float:
        return time.monotonic() - self.started_at
