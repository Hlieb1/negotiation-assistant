import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..video.face_emotion import EMOTIONS
from .sync import Observation

PROSODY_SCALES = {"loudness_db": 2.0, "pitch_st": 1.5, "f0_std_st": 0.5, "speech_rate": 0.5, "pause_ratio": 0.1}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ProsodyBaseline:
    def __init__(self, history: int = 30, min_samples: int = 3):
        self.min_samples = min_samples
        self._values: dict[str, deque] = {key: deque(maxlen=history) for key in PROSODY_SCALES}
        self._f0: deque = deque(maxlen=history)

    @property
    def ready(self) -> bool:
        return len(self._f0) >= self.min_samples

    def update(self, features: dict) -> None:
        if not features.get("f0_mean_hz"):
            return
        self._f0.append(features["f0_mean_hz"])
        for key, value in self._measures(features).items():
            self._values[key].append(value)

    def _measures(self, features: dict) -> dict[str, float]:
        f0_median = float(np.median(self._f0)) if self._f0 else features["f0_mean_hz"]
        return {
            "loudness_db": features["loudness_db"],
            "pitch_st": 12 * math.log2(features["f0_mean_hz"] / f0_median),
            "f0_std_st": features["f0_std_st"],
            "speech_rate": features["speech_rate"],
            "pause_ratio": features["pause_ratio"],
        }

    def zscores(self, features: dict) -> dict[str, float]:
        if not self.ready or not features.get("f0_mean_hz"):
            return {}
        result = {}
        for key, value in self._measures(features).items():
            history = np.array(self._values[key])
            median = float(np.median(history))
            spread = max(1.4826 * float(np.median(np.abs(history - median))), PROSODY_SCALES[key])
            result[key] = float(np.clip((value - median) / spread, -3, 3))
        return result


@dataclass
class FusedEmotion:
    probs: np.ndarray
    valence: float
    arousal: float
    hesitation: float
    modalities: list[str]
    coverage: float
    prosody_z: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {e: round(float(p), 4) for e, p in zip(EMOTIONS, self.probs)}


class LateFusion:
    def __init__(self, emotion_weights: dict[str, float], valence_weights: dict[str, float],
                 dimensions: dict[str, dict[str, float]], prosody_arousal_weight: float = 0.5):
        self.emotion_weights = emotion_weights
        self.valence_weights = valence_weights
        self.valence = np.array([dimensions[e]["valence"] for e in EMOTIONS])
        self.arousal = np.array([dimensions[e]["arousal"] for e in EMOTIONS])
        self.prosody_arousal_weight = prosody_arousal_weight

    @staticmethod
    def _weighted(sources: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        total = sum(weights[name] for name in sources)
        return sum(weights[name] * value for name, value in sources.items()) / total

    def fuse(self, observation: Observation, prosody_z: dict[str, float]) -> FusedEmotion | None:
        sources = {name: getattr(observation, name).probs for name in observation.modalities}
        if not sources:
            return None
        probs = self._weighted(sources, self.emotion_weights)
        valence = float(self._weighted({k: float(v @ self.valence) for k, v in sources.items()},
                                       self.valence_weights))
        arousal = float(probs @ self.arousal)
        hesitation = 0.0
        if prosody_z:
            excitement = np.mean([prosody_z[k] for k in ("loudness_db", "pitch_st", "f0_std_st", "speech_rate")])
            arousal = (1 - self.prosody_arousal_weight) * arousal + self.prosody_arousal_weight * sigmoid(excitement)
            hesitation = max(0.0, 2 * sigmoid(prosody_z["pause_ratio"]) - 1)
        coverage = sum(self.emotion_weights[name] for name in sources) / sum(self.emotion_weights.values())
        return FusedEmotion(probs=probs, valence=valence, arousal=arousal, hesitation=hesitation,
                            modalities=list(sources), coverage=coverage, prosody_z=prosody_z)
