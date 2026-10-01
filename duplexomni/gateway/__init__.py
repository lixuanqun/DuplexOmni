"""Full-duplex gateway: conversation in front, delegated work behind.

The paper splits a voice system into an interaction layer and a thinking
layer. This package is that split as a runnable service:

* the interaction layer keeps the floor, speaks immediately, and yields on
  barge-in;
* a delegator turns hard requests into tasks;
* a worker thinks and executes tools without blocking the audio loop;
* results stream back as fragments the interaction layer can speak.

No model weights are required. An OpenAI-compatible endpoint is optional.
"""

from .config import GatewayConfig
from .session import DuplexSession

__all__ = ["DuplexSession", "GatewayConfig"]
