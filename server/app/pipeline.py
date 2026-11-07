import numpy as np

from .protocol import Hint, Mode, now_ms


class Pipeline:
    def __init__(self, mode: Mode, hint_min_interval_s: float):
        self.mode = mode
        self.min_interval_ms = int(hint_min_interval_s * 1000)
        self.frames = 0
        self.audio_seconds = 0.0
        self._last_data_ms: int | None = None
        self._last_hint_ms = 0

    def add_video(self, frame: np.ndarray, ts: int, recv_ms: int) -> None:
        self.frames += 1
        self._last_data_ms = recv_ms

    def add_audio(self, samples: np.ndarray, sample_rate: int, ts: int, recv_ms: int) -> None:
        self.audio_seconds += len(samples) / sample_rate
        self._last_data_ms = recv_ms

    def poll(self) -> Hint | None:
        now = now_ms()
        if self._last_data_ms is None or now - self._last_hint_ms < self.min_interval_ms:
            return None
        hint = Hint(
            ts=now,
            state="нейтральний",
            confidence=0.0,
            reply="З'єднання з сервером працює.",
            tone="—",
            mode=self.mode,
            latency_ms=now - self._last_data_ms,
        )
        self._last_hint_ms = now
        self._last_data_ms = None
        return hint
