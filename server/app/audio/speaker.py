import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import SAMPLE_RATE

ME, OTHER, UNKNOWN = "me", "other", "unknown"


@dataclass
class SpeakerResult:
    label: str
    similarity: float | None = None
    processing_ms: float = 0.0


class SpeakerVerifier:
    def __init__(self, source: str, savedir: Path, profile_path: Path, device: str = "cuda:0",
                 threshold: float = 0.35, min_speech_s: float = 0.8):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.WARNING)
        try:
            self._model = EncoderClassifier.from_hparams(source=source, savedir=str(savedir),
                                                         run_opts={"device": device},
                                                         local_strategy=LocalStrategy.COPY)
        finally:
            root.setLevel(level)
        self.profile_path = profile_path
        self.threshold = threshold
        self.min_speech_s = min_speech_s
        self.profile: np.ndarray | None = np.load(profile_path) if profile_path.exists() else None

    @property
    def enrolled(self) -> bool:
        return self.profile is not None

    def warm_up(self) -> None:
        self.embed(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            vector = self._model.encode_batch(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0))
        vector = vector.squeeze().cpu().numpy()
        return vector / (np.linalg.norm(vector) + 1e-9)

    def enroll(self, speech: np.ndarray, window_s: float = 3.0) -> float:
        window = int(window_s * SAMPLE_RATE)
        parts = [speech[i:i + window] for i in range(0, max(1, len(speech) - window // 2), window)]
        vectors = [self.embed(part) for part in parts if len(part) >= SAMPLE_RATE]
        profile = np.mean(vectors, axis=0)
        self.profile = profile / np.linalg.norm(profile)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.profile_path, self.profile)
        return len(speech) / SAMPLE_RATE

    def identify(self, speech: np.ndarray) -> SpeakerResult:
        if self.profile is None or len(speech) < self.min_speech_s * SAMPLE_RATE:
            return SpeakerResult(UNKNOWN)
        started = time.perf_counter()
        similarity = float(self.embed(speech) @ self.profile)
        return SpeakerResult(ME if similarity >= self.threshold else OTHER, round(similarity, 3),
                             (time.perf_counter() - started) * 1000)
