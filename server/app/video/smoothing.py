from collections import deque
from dataclasses import dataclass

import numpy as np

from .face_emotion import EMOTIONS, FaceResult


@dataclass
class SmoothedFace:
    ts: int
    probs: np.ndarray
    face_ratio: float
    samples: int

    @property
    def dominant(self) -> str:
        return EMOTIONS[int(np.argmax(self.probs))]

    def as_dict(self) -> dict[str, float]:
        return {e: round(float(p), 4) for e, p in zip(EMOTIONS, self.probs)}


class EmotionSmoother:
    def __init__(self, window_s: float, min_face_ratio: float):
        self.window_ms = int(window_s * 1000)
        self.min_face_ratio = min_face_ratio
        self._results: deque[FaceResult] = deque()

    def add(self, result: FaceResult) -> None:
        self._results.append(result)
        while self._results and result.ts - self._results[0].ts > self.window_ms:
            self._results.popleft()

    def current(self) -> SmoothedFace | None:
        found = [r for r in self._results if r.face_found]
        if not found:
            return None
        face_ratio = len(found) / len(self._results)
        if face_ratio < self.min_face_ratio:
            return None
        return SmoothedFace(
            ts=self._results[-1].ts,
            probs=np.mean([r.probs for r in found], axis=0),
            face_ratio=face_ratio,
            samples=len(found),
        )
