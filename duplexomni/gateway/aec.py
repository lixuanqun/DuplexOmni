"""Short NLMS echo canceller.

The far end is the PCM this process just sent to the speaker. The near end
is the microphone. Weights start at zero, so an unadapted frame is unchanged.
A repeated echo of that far end is subtracted; a different near-end signal
remains.
"""

from __future__ import annotations

import array


class EchoCanceller:
    def __init__(self, order: int = 8, mu: float = 0.3) -> None:
        self.order = order
        self.mu = mu
        self._w = [0.0] * order
        self._x = [0.0] * order

    def push_far(self, pcm: bytes) -> None:
        for sample in _samples(pcm):
            self._shift(sample / 32768.0)

    def cancel(self, pcm: bytes) -> bytes:
        cleaned = array.array("h")
        for sample in _samples(pcm):
            near = sample / 32768.0
            estimate = sum(weight * value for weight, value in zip(self._w, self._x, strict=False))
            error = near - estimate
            power = sum(value * value for value in self._x) + 1e-4
            far_rms = (power / self.order) ** 0.5
            # A louder near end is the user, not a linear image of the speaker.
            if abs(near) <= far_rms * 1.2 + 0.02:
                step = self.mu * error / power
                for index, value in enumerate(self._x):
                    self._w[index] += step * value
            clipped = max(-1.0, min(1.0, error))
            cleaned.append(int(clipped * 32767))
        return cleaned.tobytes()

    def _shift(self, value: float) -> None:
        del self._x[0]
        self._x.append(value)


def _samples(pcm: bytes) -> array.array:
    usable = len(pcm) - (len(pcm) % 2)
    samples = array.array("h")
    if usable:
        samples.frombytes(pcm[:usable])
    return samples
