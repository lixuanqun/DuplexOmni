"""Collation: DatasetRecord (480 ms slices) -> Thinker/Talker training sample.

Per record we build the interaction token stream exactly as the runtime sees
it, slice by slice:

    [BOS] ˆ? [U] <user bytes> <stream events> [A] <assistant bytes> ...

with these placement rules (engineering concretisation of the paper's flat
control-token stream):

* ``ˆ`` precedes the user turn it overlaps;
* ``[THINK]`` / ``[CUT]`` / ``[WAIT]`` belong to the assistant turn and sit
  right after ``[A]`` (their ghost/spoken text split is already reflected in
  ``target_text`` which excludes ghost text);
* ``<fragment>`` and ``[PENDnS]`` are stream-level events emitted before the
  assistant turn of their slice.

The assistant span (from ``[A]`` to the end of its bytes) is both the
Thinker's prediction target and the source of conditioning tokens for the
Talker.  One sample == one dialogue; the trainer accumulates gradients over
``batch_size`` samples (paper batch 128).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.slicer import DatasetRecord
from .tokenizer import SPECIAL_TOKENS, TextTokenizer

__all__ = ["SampleTensors", "record_to_sample"]


@dataclass
class SampleTensors:
    ids: list[int]
    assistant_spans: list[tuple[int, int]]  # [start, end) per slice
    codes_per_slice: list[list[list[int]]]  # (F x K) gold assistant codes
    record_id: str = ""
    language: str = "en"
    patterns: list[str] = field(default_factory=list)


_T = TextTokenizer()


def record_to_sample(record: DatasetRecord) -> SampleTensors:
    tk = _T
    ids: list[int] = [tk.bos()]
    spans: list[tuple[int, int]] = []
    codes: list[list[list[int]]] = []

    for sl in record.slices:
        overlap = any(e["event"] == "OverlapOnset" for e in sl.events)
        if overlap:
            ids.append(SPECIAL_TOKENS["ˆ"])
        if sl.user_text:
            ids.append(SPECIAL_TOKENS["[U]"])
            ids.extend(tk.encode_text(sl.user_text))

        # stream-level events, in recorded time order
        for e in sl.events:
            name = e["event"]
            if name == "ResultFragment":
                ids.append(SPECIAL_TOKENS["<"])
                ids.extend(tk.encode_text(str(e.get("text", ""))))
                ids.append(SPECIAL_TOKENS[">"])
            elif name == "SharedSilence":
                ids.extend(tk.encode_pend(float(e.get("text", 1))))

        # assistant turn: [A] + assistant-attributed controls + spoken bytes
        start = len(ids)
        ids.append(SPECIAL_TOKENS["[A]"])
        for e in sl.events:
            name = e["event"]
            if name == "ThinkTrigger":
                ids.append(SPECIAL_TOKENS["[THINK]"])
            elif name == "Cut":
                ids.append(SPECIAL_TOKENS["[CUT]"])
            elif name == "Wait":
                ids.append(SPECIAL_TOKENS["[WAIT]"])
        if sl.target_text:
            ids.extend(tk.encode_text(sl.target_text))
        spans.append((start, len(ids)))
        codes.append(sl.assistant_codes)

    return SampleTensors(
        ids=ids,
        assistant_spans=spans,
        codes_per_slice=codes,
        record_id=record.id,
        language=record.language,
        patterns=list(record.patterns),
    )
