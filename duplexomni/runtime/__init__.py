"""Runtime: full-duplex interaction layer + pluggable thinking layer."""

from .bridge import ThinkingBridge
from .events import (
    AssistantSpeechSlice,
    AssistantTextDelta,
    BargeInDetected,
    Overlap,
    RuntimeEvent,
    SharedSilence,
    SliceOutput,
    SpeechCut,
    ThinkingAborted,
    ThinkingFragment,
    ThinkingRequested,
    summarise,
)
from .interaction import InteractionLayer, SimClock, UserSliceInput, WallClock
from .metrics import LatencyMeter, RTFMeter, percentile
from .session import SessionState
from .thinking import (
    EchoThinkingLayer,
    FunctionThinkingLayer,
    OpenAICompatThinkingLayer,
    ThinkingContext,
    ThinkingLayer,
    ThinkingRequest,
)
from .vad import EnergyVAD

__all__ = [
    "AssistantSpeechSlice",
    "AssistantTextDelta",
    "BargeInDetected",
    "EchoThinkingLayer",
    "EnergyVAD",
    "FunctionThinkingLayer",
    "InteractionLayer",
    "LatencyMeter",
    "OpenAICompatThinkingLayer",
    "Overlap",
    "RTFMeter",
    "RuntimeEvent",
    "SessionState",
    "SharedSilence",
    "SimClock",
    "SliceOutput",
    "SpeechCut",
    "ThinkingAborted",
    "ThinkingBridge",
    "ThinkingContext",
    "ThinkingFragment",
    "ThinkingLayer",
    "ThinkingRequested",
    "ThinkingRequest",
    "UserSliceInput",
    "WallClock",
    "percentile",
    "summarise",
]
