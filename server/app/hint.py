from .decision.base import Decision, StateVector
from .protocol import Hint, Mode


class HintPolicy:
    def __init__(self, min_interval_s: float, stable_s: float, min_confidence: float):
        self.min_interval_ms = int(min_interval_s * 1000)
        self.stable_ms = int(stable_s * 1000)
        self.min_confidence = min_confidence
        self._candidate: str | None = None
        self._candidate_since = 0
        self._last_state: str | None = None
        self._last_hint_ms: int | None = None

    def should_hint(self, state: StateVector, now_ms: int) -> bool:
        if state.name != self._candidate:
            self._candidate, self._candidate_since = state.name, now_ms
        if state.name == self._last_state or state.confidence < self.min_confidence:
            return False
        if now_ms - self._candidate_since < self.stable_ms:
            return False
        if self._last_hint_ms is not None and now_ms - self._last_hint_ms < self.min_interval_ms:
            return False
        self._last_state, self._last_hint_ms = state.name, now_ms
        return True


def build_hint(decision: Decision, state: StateVector, mode: Mode, now_ms: int, latency_ms: int) -> Hint:
    return Hint(
        ts=now_ms,
        state=state.name,
        confidence=round(min(1.0, max(0.0, state.confidence)), 3),
        reply=decision.reply,
        tone=decision.tone,
        mode=mode,
        latency_ms=latency_ms,
    )
