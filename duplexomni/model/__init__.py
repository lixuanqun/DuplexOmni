"""Model package (requires PyTorch; install ``duplexomni[torch]``)."""

from .code2wav import Code2Wav
from .collate import SampleTensors, record_to_sample
from .full import AlternateTrainer, DuplexOmni, SessionGeneration, train_two_stage
from .mtp import MTPHead
from .talker import Talker
from .thinker import Thinker, ThinkerOutput
from .tokenizer import SPECIAL_TOKENS, TEXT_VOCAB_SIZE, TextTokenizer
from .transformer import KVCache

__all__ = [
    "AlternateTrainer",
    "Code2Wav",
    "DuplexOmni",
    "KVCache",
    "MTPHead",
    "SPECIAL_TOKENS",
    "SampleTensors",
    "SessionGeneration",
    "Talker",
    "TEXT_VOCAB_SIZE",
    "TextTokenizer",
    "Thinker",
    "ThinkerOutput",
    "record_to_sample",
    "train_two_stage",
]
