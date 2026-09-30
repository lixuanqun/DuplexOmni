"""Consistency checks for annotated dialogs (paper Sec. 4.1 filtering step).

The paper filters Director output through consistency checks before TTS and
time-slicing.  Here each rule returns errors (reject sample) or warnings
(keep, but log).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tokens import (
    Cut,
    GhostText,
    OverlapOnset,
    ResultFragment,
    SharedSilence,
    Speaker,
    TextDelta,
    ThinkTrigger,
    TurnStart,
    Wait,
    parse_script,
)
from .director import AnnotatedDialog, UtteranceKind
from .scenario import Pattern

__all__ = ["CheckResult", "check_annotated_dialog", "filter_dialogs"]


@dataclass
class CheckResult:
    scenario_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_annotated_dialog(dialog: AnnotatedDialog) -> CheckResult:
    res = CheckResult(scenario_id=dialog.seed.scenario_id)
    script = dialog.to_script()
    events = parse_script(script)

    _check_turn_structure(dialog, events, res)
    _check_think_flow(events, res)
    _check_cut_ghost(events, res)
    _check_overlap_placement(dialog, events, res)
    _check_silence(dialog, res)
    _check_patterns_instantiated(dialog, res)
    return res


def _check_turn_structure(dialog: AnnotatedDialog, events: list, res: CheckResult) -> None:
    speakers = {u.speaker for u in dialog.utterances if u.kind == UtteranceKind.SPEECH}
    if Speaker.USER not in speakers:
        res.error("no user utterance found")
    if Speaker.ASSISTANT not in speakers:
        res.error("no assistant utterance found")
    starts = [e for e in events if isinstance(e, TurnStart)]
    if not starts:
        res.error("script has no speaker tags")
    # text sanity: every speech utterance must carry text or control tokens
    for u in dialog.utterances:
        if u.kind != UtteranceKind.SPEECH:
            continue
        utt_events = parse_script(u.text)
        stripped = "".join(
            e.text for e in utt_events if isinstance(e, (TextDelta, GhostText))
        ).strip()
        has_control = any(
            isinstance(e, (ThinkTrigger, Cut, Wait, OverlapOnset, SharedSilence, ResultFragment))
            for e in utt_events
        )
        if not stripped and not has_control:
            res.error(f"speech utterance with no text: {u.text[:40]!r}")
        if u.duration_s <= 0:
            res.error(f"non-positive duration for utterance: {u.text[:40]!r}")


def _check_think_flow(events: list, res: CheckResult) -> None:
    """Think-request lifecycle: [THINK] -> fragments -> optional [WAIT] abort.

    A fragment without an open [THINK] request is an orphan result; a [THINK]
    that never receives a fragment nor an abort is an unresolved request.
    """
    pending_think = False
    for ev in events:
        if isinstance(ev, ThinkTrigger):
            if pending_think:
                res.warn("two [THINK] triggers without an intervening resolution")
            pending_think = True
        elif isinstance(ev, ResultFragment):
            if not pending_think:
                res.error("orphan result fragment: no open [THINK] request")
            pending_think = True  # further fragments of the same request
        elif isinstance(ev, Wait):
            pending_think = False
    if pending_think:
        # a request left open with no fragment and no abort
        if not any(isinstance(e, ResultFragment) for e in events):
            res.error("[THINK] never resolved: no <...> fragment and no [WAIT] reset")


def _check_cut_ghost(events: list, res: CheckResult) -> None:
    """[CUT] should preserve ghost text in the same utterance (paper App. A)."""
    saw_cut = False
    ghost = False
    for ev in events:
        if isinstance(ev, Cut):
            if saw_cut and not ghost:
                res.warn("[CUT] without ghost text before the next [CUT]")
            saw_cut = True
            ghost = False
        elif isinstance(ev, GhostText) and ev.text.strip():
            ghost = True
        elif isinstance(ev, TurnStart):
            if saw_cut and not ghost:
                res.warn("[CUT] at end of utterance with no ghost text preserved")
            saw_cut, ghost = False, False
    if saw_cut and not ghost:
        res.warn("[CUT] at end of script with no ghost text preserved")


def _check_overlap_placement(dialog: AnnotatedDialog, events: list, res: CheckResult) -> None:
    for u in dialog.utterances:
        if u.backchannel and not u.overlap:
            res.error("backchannel utterance must overlap assistant speech")
        if u.speaker is Speaker.ASSISTANT and u.overlap:
            res.warn("assistant marked as overlapping: unusual for this grammar")
    overlaps = [e for e in events if isinstance(e, OverlapOnset)]
    if not overlaps:
        return
    user_overlap_text = sum(
        1 for u in dialog.utterances if u.overlap and u.speaker is Speaker.USER
    )
    if user_overlap_text == 0:
        res.error("overlap tokens present but no user utterance is marked overlap=True")


def _check_silence(dialog: AnnotatedDialog, res: CheckResult) -> None:
    for u in dialog.utterances:
        if u.kind == UtteranceKind.SILENCE:
            text = u.text.strip()
            if not (text.startswith("[PEND") and text.endswith("S]")):
                res.error(f"malformed shared-silence token: {text!r}")
            else:
                try:
                    n = float(text[5:-2])
                    if not 0.5 <= n <= 10.0:
                        res.warn(f"shared silence out of comfortable range: {n}s")
                except ValueError:
                    res.error(f"unparsable shared-silence duration: {text!r}")


def _check_patterns_instantiated(dialog: AnnotatedDialog, res: CheckResult) -> None:
    script = dialog.to_script()
    events = parse_script(script)
    instantiated: set[str] = set()
    if any(isinstance(e, ThinkTrigger) for e in events):
        instantiated.add(Pattern.DELAYED_REASONING.value)
    if any(isinstance(e, ResultFragment) for e in events):
        instantiated.add(Pattern.DELAYED_REASONING.value)
    if any(isinstance(e, SharedSilence) for e in events):
        instantiated.add(Pattern.SHARED_SILENCE.value)
    if any(isinstance(e, OverlapOnset) for e in events):
        instantiated.add(Pattern.OVERLAP.value)
    if any(isinstance(e, Cut) for e in events):
        instantiated.add(Pattern.INTERRUPTION_RESET.value)
    if dialog.utterances and dialog.utterances[0].speaker is Speaker.ASSISTANT:
        instantiated.add(Pattern.ASSISTANT_INITIATED.value)
    if any(u.backchannel for u in dialog.utterances):
        instantiated.add(Pattern.BACKCHANNEL.value)

    for p in dialog.seed.patterns:
        if p.value not in instantiated:
            res.warn(f"requested pattern not instantiated: {p.value}")
    if not instantiated:
        res.error("no interaction pattern instantiated at all")


def filter_dialogs(
    dialogs: list[AnnotatedDialog],
) -> tuple[list[AnnotatedDialog], list[CheckResult], dict[str, float]]:
    """Run checks and split into (kept, results, stats)."""
    kept: list[AnnotatedDialog] = []
    results: list[CheckResult] = []
    for d in dialogs:
        r = check_annotated_dialog(d)
        results.append(r)
        if r.ok:
            kept.append(d)
    stats = {
        "n_total": len(dialogs),
        "n_kept": len(kept),
        "keep_rate": len(kept) / max(len(dialogs), 1),
        "n_errors": sum(len(r.errors) for r in results),
        "n_warnings": sum(len(r.warnings) for r in results),
    }
    return kept, results, stats
