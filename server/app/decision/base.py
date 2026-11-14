from abc import ABC, abstractmethod
from dataclasses import dataclass, field

STATE_KEYS = {"злість": "anger", "напруга": "tension", "сумнів": "doubt", "згода": "agreement",
              "нейтральний": "neutral"}
STRATEGIES = ("calm", "argue", "clarify", "close", "pause")
STRATEGIES_UK = {
    "calm": "заспокоїти",
    "argue": "аргументувати",
    "clarify": "уточнити",
    "close": "закріпити домовленість",
    "pause": "пауза",
}


@dataclass
class StateVector:
    name: str
    degrees: dict[str, float]
    confidence: float
    valence: float = 0.0
    arousal: float = 0.5
    hesitation: float = 0.0


@dataclass
class Decision:
    reply: str
    tone: str
    confidence: float
    source: str
    strategy: str | None = None
    details: dict = field(default_factory=dict)


class DecisionModule(ABC):
    name: str

    @abstractmethod
    def decide(self, state: StateVector, voice_features: dict, transcript: str,
               history: list[Decision], context: str = "") -> Decision:
        ...

    def warm_up(self) -> None:
        pass
