"""Interaction-floor state machine.

Speech ownership (idle / user / assistant) is independent from whether a
delegated task is running. That is the paper's split: the assistant may
talk while work proceeds, and a barge-in both cuts speech and aborts the
task. The machine is pure — the session applies the commands it returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Floor(str, Enum):
    IDLE = "idle"
    USER = "user"
    ASSISTANT = "assistant"


class ThinkState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"


@dataclass(frozen=True)
class Command:
    name: str
    data: dict = field(default_factory=dict)


class FloorController:
    def __init__(self) -> None:
        self.floor = Floor.IDLE
        self.thinking = ThinkState.IDLE
        self.turn_id = 0
        self.task_id: str | None = None
        self._hold_until: float | None = None
        self._play_cursor: float | None = None
        self._client_playing = False

    def user_voice(self, *, now: float) -> list[Command]:
        """Mark the user as holding the floor. Barge-in is ``preempt``, not this."""
        if self.floor == Floor.USER:
            return []
        self.floor = Floor.USER
        self._hold_until = None
        self._play_cursor = None
        return [Command("floor", {"floor": self.floor.value, "at": now})]

    def user_release(self, *, now: float) -> list[Command]:
        if self.floor != Floor.USER:
            return []
        self.floor = Floor.IDLE
        return [Command("floor", {"floor": self.floor.value, "at": now})]

    def begin_speech(self, text: str, hold_s: float, *, now: float) -> list[Command]:
        if self.floor == Floor.USER:
            return [Command("defer_speech", {"text": text})]
        if self.floor == Floor.ASSISTANT:
            return self.extend_speech(text, hold_s, now=now)
        self.turn_id += 1
        self.floor = Floor.ASSISTANT
        self._hold_until = now + max(hold_s, 0.0)
        return [
            Command("speak", {"text": text, "turn_id": self.turn_id, "hold_s": hold_s}),
            Command("floor", {"floor": self.floor.value, "turn_id": self.turn_id}),
        ]

    def extend_speech(self, text: str, hold_s: float, *, now: float) -> list[Command]:
        if self.floor == Floor.USER:
            return [Command("defer_speech", {"text": text})]
        if self.floor != Floor.ASSISTANT:
            return self.begin_speech(text, hold_s, now=now)
        base = self._hold_until if self._hold_until and self._hold_until > now else now
        self._hold_until = min(base + max(hold_s, 0.0), now + 20.0)
        return [Command("speak", {"text": text, "turn_id": self.turn_id, "hold_s": hold_s})]

    def note_sent_frame(self, *, now: float, frame_s: float) -> None:
        """Push the hold out to the moment this outbound frame finishes playing."""
        if self.floor != Floor.ASSISTANT:
            return
        cursor = self._play_cursor if self._play_cursor is not None and self._play_cursor > now else now
        self._play_cursor = cursor + max(frame_s, 0.0)
        if self._hold_until is None or self._play_cursor > self._hold_until:
            self._hold_until = self._play_cursor

    def server_playing(self, now: float) -> bool:
        return self._play_cursor is not None and now < self._play_cursor

    def note_playback(self, active: bool, *, now: float) -> list[Command]:
        """Hold the floor while the client is actually speaking.

        The estimated hold is only a fallback for clients that never report
        playback. A start stretches the hold; an end releases it on the next tick.
        """
        if self.floor != Floor.ASSISTANT:
            self._client_playing = False
            return []
        self._client_playing = active
        if active:
            self._hold_until = now + 30.0
        else:
            release = now + 0.12
            if self._play_cursor is not None and self._play_cursor > release:
                self._hold_until = self._play_cursor
            else:
                self._hold_until = release
        return []

    def tick(self, *, now: float) -> list[Command]:
        if self.floor == Floor.ASSISTANT and self._hold_until is not None and now >= self._hold_until:
            self.floor = Floor.IDLE
            self._hold_until = None
            self._play_cursor = None
            self._client_playing = False
            return [Command("floor", {"floor": self.floor.value})]
        return []

    def preempt(self, reason: str) -> list[Command]:
        """Drop assistant audio and any in-flight task. Used on barge-in and new turns."""
        cmds: list[Command] = []
        self._client_playing = False
        self._play_cursor = None
        if self.floor == Floor.ASSISTANT:
            self.floor = Floor.IDLE
            self._hold_until = None
            cmds.append(Command("cut", {"reason": reason}))
            cmds.append(Command("floor", {"floor": self.floor.value}))
        cmds.extend(self.cancel_thinking(reason))
        return cmds

    def start_thinking(self, task_id: str) -> list[Command]:
        cmds: list[Command] = []
        if self.thinking == ThinkState.RUNNING and self.task_id and self.task_id != task_id:
            cmds.append(Command("cancel_task", {"reason": "superseded", "task_id": self.task_id}))
        self.thinking = ThinkState.RUNNING
        self.task_id = task_id
        cmds.append(Command("start_task", {"task_id": task_id}))
        return cmds

    def finish_thinking(self, task_id: str) -> list[Command]:
        if self.task_id != task_id or self.thinking != ThinkState.RUNNING:
            return []
        self.thinking = ThinkState.IDLE
        self.task_id = None
        return [Command("task_idle", {"task_id": task_id})]

    def cancel_thinking(self, reason: str) -> list[Command]:
        if self.thinking != ThinkState.RUNNING:
            return []
        task_id = self.task_id
        self.thinking = ThinkState.IDLE
        self.task_id = None
        return [Command("cancel_task", {"reason": reason, "task_id": task_id})]
