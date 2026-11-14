from dataclasses import dataclass
from pathlib import Path

import yaml

from ..video.face_emotion import EMOTIONS
from .fusion import FusedEmotion

SIGNALS = ("negativity", "positivity", "arousal_high", "arousal_low", "hesitation")


@dataclass
class NegotiationState:
    name: str
    degrees: dict[str, float]
    confidence: float
    fused: FusedEmotion


def load_rules(path: Path) -> dict:
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, state in rules["states"].items():
        unknown = (set(state.get("emotions", {})) - set(EMOTIONS)) | (set(state.get("signals", {})) - set(SIGNALS))
        if unknown:
            raise ValueError(f"state '{name}' uses unknown terms: {sorted(unknown)}")
    return rules


def signals(fused: FusedEmotion) -> dict[str, float]:
    return {
        "negativity": max(0.0, -fused.valence),
        "positivity": max(0.0, fused.valence),
        "arousal_high": max(0.0, fused.arousal - 0.5) * 2,
        "arousal_low": max(0.0, 0.5 - fused.arousal) * 2,
        "hesitation": fused.hesitation,
    }


class StateStabilizer:
    def __init__(self, margin: float):
        self.margin = margin
        self.current: str | None = None

    def update(self, state: NegotiationState) -> NegotiationState:
        degrees = state.degrees
        if self.current is None or self.current not in degrees or \
                degrees[state.name] - degrees[self.current] >= self.margin:
            self.current = state.name
            return state
        if state.name == self.current:
            return state
        uniform = 1.0 / len(degrees)
        sharpness = max(0.0, (degrees[self.current] - uniform) / (1.0 - uniform))
        confidence = sharpness * (0.6 + 0.4 * state.fused.coverage)
        return NegotiationState(name=self.current, degrees=degrees, confidence=round(confidence, 4),
                                fused=state.fused)


class StateMapper:
    def __init__(self, rules: dict):
        self.states = rules["states"]

    def map(self, fused: FusedEmotion) -> NegotiationState:
        probs = dict(zip(EMOTIONS, fused.probs))
        values = signals(fused)
        scores = {
            name: sum(w * probs[e] for e, w in state.get("emotions", {}).items())
            + sum(w * values[s] for s, w in state.get("signals", {}).items())
            for name, state in self.states.items()
        }
        total = sum(scores.values()) or 1.0
        degrees = {name: score / total for name, score in scores.items()}
        name = max(degrees, key=degrees.get)
        uniform = 1.0 / len(degrees)
        sharpness = (degrees[name] - uniform) / (1.0 - uniform)
        confidence = sharpness * (0.6 + 0.4 * fused.coverage)
        return NegotiationState(name=name, degrees={k: round(v, 4) for k, v in degrees.items()},
                                confidence=round(confidence, 4), fused=fused)
