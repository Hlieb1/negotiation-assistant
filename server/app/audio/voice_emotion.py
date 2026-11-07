import logging
import time
from dataclasses import dataclass

import numpy as np

from ..video.face_emotion import EMOTIONS
from .preprocessing import SAMPLE_RATE

EMOTION2VEC_LABELS = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprise",
}


@dataclass
class VoiceResult:
    ts: int
    probs: np.ndarray | None
    unknown: float = 0.0
    processing_ms: float = 0.0


def to_emotion_vector(labels: list[str], scores: list[float]) -> tuple[np.ndarray, float]:
    probs = np.zeros(len(EMOTIONS))
    unknown = 0.0
    for label, score in zip(labels, scores):
        name = label.split("/")[-1]
        if name in EMOTION2VEC_LABELS:
            probs[EMOTIONS.index(EMOTION2VEC_LABELS[name])] += score
        else:
            unknown += score
    total = probs.sum()
    return (probs / total if total > 0 else probs), float(unknown)


class VoiceEmotionClassifier:
    def __init__(self, model: str = "emotion2vec/emotion2vec_plus_large", device: str = "cuda:0",
                 min_speech_s: float = 0.5):
        from funasr import AutoModel

        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.WARNING)
        try:
            self._model = AutoModel(model=model, hub="hf", device=device, disable_update=True, log_level="ERROR")
        finally:
            root.setLevel(level)
        self.min_speech_s = min_speech_s

    def warm_up(self) -> None:
        self._model.generate(np.zeros(SAMPLE_RATE, dtype=np.float32), granularity="utterance",
                             extract_embedding=False, disable_pbar=True)

    def predict(self, speech: np.ndarray, ts: int) -> VoiceResult:
        started = time.perf_counter()
        if len(speech) < self.min_speech_s * SAMPLE_RATE:
            return VoiceResult(ts=ts, probs=None)
        output = self._model.generate(speech.astype(np.float32), granularity="utterance",
                                      extract_embedding=False, disable_pbar=True)[0]
        probs, unknown = to_emotion_vector(output["labels"], output["scores"])
        return VoiceResult(ts=ts, probs=probs, unknown=unknown,
                           processing_ms=(time.perf_counter() - started) * 1000)
