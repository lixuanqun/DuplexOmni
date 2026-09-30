"""Scripted behavioural scenarios mirroring Full DuplexBench (paper Sec. 5).

Each scenario scripts the user side (which slices carry speech + ASR text)
and asserts *behavioural* properties of the interaction layer: when it may
speak, when it must cut, that async thinking never blocks the slice loop,
and that an interruption resets in-flight reasoning.

The model behind the runtime is untrained in this repository, so — honestly —
these checks validate the **runtime policy and collaboration machinery**, not
language quality.  They are the systems-level analogue of the paper's
DuplexBench categories (interruption, overlap, backchannel, silence).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..runtime import SliceOutput
from ..runtime.events import BargeInDetected, SpeechCut, ThinkingAborted, ThinkingFragment

__all__ = [
    "UserTurn",
    "Scenario",
    "default_scenarios",
    "policy_scenarios",
    "thinking_scenarios",
    "floor_discipline",
    "turn_taking",
    "thinking_delivery",
    "thinking_abort",
]


@dataclass(frozen=True)
class UserTurn:
    slice_index: int
    text: str
    loud: bool = True  # False = inaudible (VAD misses it)


@dataclass
class Scenario:
    name: str
    category: str
    n_slices: int
    turns: list[UserTurn]
    check: Callable[[list[SliceOutput], Scenario], list[str]]
    thinking: dict | None = None   # EchoThinkingLayer kwargs -> seed a request at slice 0
    requires_thinking: bool = False
    description: str = ""


def _events_of(outputs: list[SliceOutput], kind) -> list[tuple[int, object]]:
    return [(o.index, e) for o in outputs for e in o.events if isinstance(e, kind)]


def _talk_over_slices(outputs: list[SliceOutput]) -> list[int]:
    return [o.index for o in outputs if o.user_speaking and o.assistant_speaking]


def _rtf_violations(outputs: list[SliceOutput]) -> list[str]:
    return [
        f"rtf={o.rtf:.3f} at slice {o.index}" for o in outputs if o.rtf >= 1.0
    ]


# --------------------------------------------------------------------------- #
# Scenario factories
# --------------------------------------------------------------------------- #


def floor_discipline() -> Scenario:
    """The user holds the floor: the assistant must never talk over them,
    and a genuine barge-in must cut the current turn exactly once."""

    def check(outputs: list[SliceOutput], sc: Scenario) -> list[str]:
        v = _rtf_violations(outputs)
        over = _talk_over_slices(outputs)
        if over:
            v.append(f"assistant talked over the user at slices {over}")
        barge = [i for i, _ in _events_of(outputs, BargeInDetected)]
        if not barge:
            v.append("expected a barge-in when the user spoke over an open turn")
        cuts = [i for i, _ in _events_of(outputs, SpeechCut)]
        if len(cuts) < len(barge):
            v.append(f"only {len(cuts)} SpeechCut for {len(barge)} barge-ins")
        return v

    return Scenario(
        name="floor-discipline",
        category="interruption",
        n_slices=12,
        turns=[
            UserTurn(1, "wait, let me say something first"),
            UserTurn(3, "and one more thing about the schedule"),
        ],
        check=check,
        description="barge-in cuts the turn; assistant stays silent while the user speaks",
    )


def turn_taking(respond_within: int = 5) -> Scenario:
    """After the user stops speaking, the assistant must take its turn
    within ``respond_within`` slices."""

    def check(outputs: list[SliceOutput], sc: Scenario) -> list[str]:
        v = _rtf_violations(outputs)
        over = _talk_over_slices(outputs)
        if over:
            v.append(f"assistant talked over the user at slices {over}")
        last_user = max(t.slice_index for t in sc.turns)
        window = range(last_user + 1, min(last_user + 1 + respond_within, len(outputs)))
        spoke = [i for i in window if outputs[i].assistant_speaking]
        if not spoke:
            v.append(
                f"assistant did not respond within {respond_within} slices "
                f"after the user's last turn at slice {last_user}"
            )
        return v

    return Scenario(
        name="turn-taking",
        category="turn_taking",
        n_slices=12,
        turns=[UserTurn(3, "so what is the plan for tomorrow then")],
        check=check,
        description=f"assistant responds within {respond_within} slices after the user stops",
    )


def thinking_delivery(n_fragments: int = 3) -> Scenario:
    """A thinking request seeded at session start must deliver its fragments
    asynchronously without blocking the slice loop (delayed reasoning)."""

    def check(outputs: list[SliceOutput], sc: Scenario) -> list[str]:
        v = _rtf_violations(outputs)
        frags = _events_of(outputs, ThinkingFragment)
        if not frags:
            v.append("no thinking fragment was delivered")
        if not any(o.assistant_speaking for o in outputs):
            v.append("interaction layer never spoke while thinking was in flight")
        return v

    return Scenario(
        name="thinking-delivery",
        category="delayed_reasoning",
        n_slices=12,
        # slice 3: after the assistant's opening turn has closed, so the user
        # speech does NOT barge in and cancel the in-flight request
        turns=[UserTurn(3, "can you check the numbers for me")],
        check=check,
        thinking={"delay_s": 0.0, "n_fragments": n_fragments},
        requires_thinking=True,
        description="[THINK] fragments stream in while the interaction layer keeps talking",
    )


def thinking_abort(n_fragments: int = 12, interrupt_at: int = 1) -> Scenario:
    """Interrupting while a reasoning request is in flight must abort it
    ([WAIT] semantics): no fragment may be injected after the abort."""

    def check(outputs: list[SliceOutput], sc: Scenario) -> list[str]:
        v = _rtf_violations(outputs)
        aborts = [i for i, _ in _events_of(outputs, ThinkingAborted)]
        if not aborts:
            v.append("in-flight thinking was not aborted by the interruption")
            return v
        abort_slice = min(aborts)
        late = [i for i, _ in _events_of(outputs, ThinkingFragment) if i > abort_slice]
        if late:
            v.append(f"fragments delivered after the abort at slice {abort_slice}: {late}")
        barge = [i for i, _ in _events_of(outputs, BargeInDetected)]
        if interrupt_at not in barge:
            v.append(f"expected the barge-in at slice {interrupt_at}")
        return v

    return Scenario(
        name="thinking-abort",
        category="interruption_reset",
        n_slices=12,
        turns=[UserTurn(interrupt_at, "wait wait, forget that question")],
        check=check,
        thinking={"delay_s": 0.02, "n_fragments": n_fragments},
        requires_thinking=True,
        description="[WAIT]: interruption cancels the pending request; late fragments are dropped",
    )


def policy_scenarios() -> list[Scenario]:
    """Scenarios that do not require a thinking layer (usable with bridge=None)."""
    return [floor_discipline(), turn_taking()]


def thinking_scenarios() -> list[Scenario]:
    return [thinking_delivery(), thinking_abort()]


def default_scenarios() -> list[Scenario]:
    return policy_scenarios() + thinking_scenarios()
