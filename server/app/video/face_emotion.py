import time
from dataclasses import dataclass

import numpy as np

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EMOTIONS_UK = {
    "angry": "злість",
    "disgust": "відраза",
    "fear": "страх",
    "happy": "радість",
    "sad": "смуток",
    "surprise": "здивування",
    "neutral": "нейтральний",
}


@dataclass
class FaceResult:
    ts: int
    probs: np.ndarray | None
    box: tuple[int, int, int, int] | None = None
    face_confidence: float = 0.0
    processing_ms: float = 0.0

    @property
    def face_found(self) -> bool:
        return self.probs is not None


class FaceEmotionRecognizer:
    def __init__(self, detector_backend: str = "opencv", align: bool = True, min_face_size: int = 40):
        from deepface import DeepFace

        self._deepface = DeepFace
        self.detector_backend = detector_backend
        self.align = align
        self.min_face_size = min_face_size

    def warm_up(self) -> None:
        self._deepface.analyze(np.zeros((480, 640, 3), dtype=np.uint8), actions=["emotion"],
                               detector_backend=self.detector_backend, enforce_detection=False, silent=True)

    def analyze(self, frame: np.ndarray, ts: int) -> FaceResult:
        started = time.perf_counter()
        try:
            faces = self._deepface.analyze(frame, actions=["emotion"], detector_backend=self.detector_backend,
                                           align=self.align, enforce_detection=True, silent=True)
        except ValueError:
            faces = []
        faces = [f for f in faces if min(f["region"]["w"], f["region"]["h"]) >= self.min_face_size]
        elapsed = (time.perf_counter() - started) * 1000
        if not faces:
            return FaceResult(ts=ts, probs=None, processing_ms=elapsed)
        face = max(faces, key=lambda f: f["region"]["w"] * f["region"]["h"])
        probs = np.array([face["emotion"][e] for e in EMOTIONS], dtype=np.float64)
        region = face["region"]
        return FaceResult(
            ts=ts,
            probs=probs / probs.sum(),
            box=(region["x"], region["y"], region["w"], region["h"]),
            face_confidence=float(face.get("face_confidence") or 0.0),
            processing_ms=elapsed,
        )
