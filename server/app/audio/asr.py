import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import SAMPLE_RATE


def add_nvidia_dll_dirs() -> None:
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if os.name != "nt" or not nvidia.exists():
        return
    for bin_dir in nvidia.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


@dataclass
class Transcript:
    ts: int
    text: str
    duration_s: float
    processing_ms: float


class SpeechRecognizer:
    def __init__(self, model: str = "small", language: str = "uk", device: str = "cuda",
                 compute_type: str = "float16", beam_size: int = 1):
        add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    def warm_up(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), 0)

    def transcribe(self, audio: np.ndarray, ts: int) -> Transcript:
        started = time.perf_counter()
        segments, _ = self._model.transcribe(audio.astype(np.float32), language=self.language,
                                             beam_size=self.beam_size, condition_on_previous_text=False,
                                             vad_filter=False)
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
        return Transcript(ts=ts, text=text, duration_s=round(len(audio) / SAMPLE_RATE, 2),
                          processing_ms=(time.perf_counter() - started) * 1000)


class UtteranceBuffer:
    def __init__(self, min_pause_s: float = 0.6, max_utterance_s: float = 10.0):
        self.min_pause_s = min_pause_s
        self.max_samples = int(max_utterance_s * SAMPLE_RATE)
        self._chunks: list[np.ndarray] = []
        self._start_ts: int | None = None

    def add(self, speech: np.ndarray, trailing_silence_s: float, ts: int) -> tuple[np.ndarray, int] | None:
        if len(speech):
            if self._start_ts is None:
                self._start_ts = ts
            self._chunks.append(speech)
        size = sum(len(c) for c in self._chunks)
        if size and (trailing_silence_s >= self.min_pause_s or size >= self.max_samples):
            return self.flush()
        return None

    def flush(self) -> tuple[np.ndarray, int] | None:
        if not self._chunks:
            return None
        utterance = (np.concatenate(self._chunks), self._start_ts)
        self._chunks, self._start_ts = [], None
        return utterance
