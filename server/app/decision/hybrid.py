import time

from .base import Decision, DecisionModule, StateVector
from .fuzzy import FuzzyDecision
from .llm import LLMDecision, LLMError, describe_situation


class HybridDecision(DecisionModule):
    name = "hybrid"

    def __init__(self, fuzzy: FuzzyDecision, llm: LLMDecision):
        self.fuzzy = fuzzy
        self.llm = llm

    def decide(self, state: StateVector, voice_features: dict, transcript: str,
               history: list[Decision]) -> Decision:
        base = self.fuzzy.decide(state, voice_features, transcript, history)
        started = time.perf_counter()
        situation = describe_situation(state, voice_features, transcript, history, base.strategy, base.tone)
        try:
            answer = self.llm.generate(situation)
        except LLMError as exc:
            base.source = f"{self.name}_fallback"
            base.details.update(error=str(exc), llm_ms=round((time.perf_counter() - started) * 1000))
            return base
        base.details["llm_ms"] = round((time.perf_counter() - started) * 1000)
        return Decision(reply=answer.reply, tone=base.tone, confidence=state.confidence, source=self.name,
                        strategy=base.strategy, details=base.details)
