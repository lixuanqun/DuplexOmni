"""Byte-level text tokenizer with full-duplex control special tokens.

The paper uses the Qwen tokenizer over the Thinker's MLLM backbone; for this
reference implementation a UTF-8 byte tokenizer (ids 0..255) plus one special
id per control token keeps everything self-contained and language-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TextTokenizer", "N_SPECIAL", "TEXT_VOCAB_SIZE", "SPECIAL_TOKENS"]

SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 256,
    "[U]": 257,
    "[A]": 258,
    "[THINK]": 259,
    "[CUT]": 260,
    "[WAIT]": 261,
    "ˆ": 262,  # overlap onset
    "[PEND": 263,
    "]": 264,
    "<": 265,
    ">": 266,
    "[BOS]": 267,
    "[EOS]": 268,
}

N_SPECIAL = len(SPECIAL_TOKENS)
TEXT_VOCAB_SIZE = 256 + N_SPECIAL

_PAD = SPECIAL_TOKENS["<pad>"]
_BOS = SPECIAL_TOKENS["[BOS]"]
_EOS = SPECIAL_TOKENS["[EOS]"]


@dataclass
class EncodedStream:
    ids: list[int]
    # which ids belong to *predictable* positions (assistant-side tokens the
    # Thinker is trained on with next-token loss)
    target_mask: list[bool]

    def __len__(self) -> int:
        return len(self.ids)


class TextTokenizer:
    """UTF-8 bytes + control specials; control characters survive round-trip."""

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_control(self, name: str) -> int:
        return SPECIAL_TOKENS[name]

    def encode_pend(self, seconds: float) -> list[int]:
        return [SPECIAL_TOKENS["[PEND"], *self.encode_text(f"{seconds:g}S"), SPECIAL_TOKENS["]"]]

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf = bytearray()
        reverse = {v: k for k, v in SPECIAL_TOKENS.items()}
        for i in ids:
            if i < 256:
                buf.append(i)
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(reverse.get(i, f"<{i}>"))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def bos(self) -> int:
        return _BOS

    def eos(self) -> int:
        return _EOS

    def pad(self) -> int:
        return _PAD
