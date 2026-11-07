from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .face_emotion import EMOTIONS


class TimedProbs(Protocol):
    ts: int
    probs: np.ndarray | None


@dataclass
class SmoothedEmotion:
    ts: int
    probs: np.ndarray
    found_ratio: float
    samples: int

    @property
    def dominant(self) -> str:
        return EMOTIONS[int(np.argmax(self.probs))]

    def as_dict(self) -> dict[str, float]:
        return {e: round(float(p), 4) for e, p in zip(EMOTIONS, self.probs)}


class EmotionSmoother:
    def __init__(self, window_s: float, min_found_ratio: float = 0.0):
        self.window_ms = int(window_s * 1000)
        self.min_found_ratio = min_found_ratio
        self._results: deque[TimedProbs] = deque()

    def add(self, result: TimedProbs) -> None:
        self._results.append(result)
        while self._results and result.ts - self._results[0].ts > self.window_ms:
            self._results.popleft()

    def current(self) -> SmoothedEmotion | None:
        found = [r for r in self._results if r.probs is not None]
        if not found:
            return None
        found_ratio = len(found) / len(self._results)
        if found_ratio < self.min_found_ratio:
            return None
        return SmoothedEmotion(
            ts=self._results[-1].ts,
            probs=np.mean([r.probs for r in found], axis=0),
            found_ratio=found_ratio,
            samples=len(found),
        )
