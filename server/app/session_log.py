import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .protocol import now_ms


class SessionLog:
    def __init__(self, log_dir: Path, session_id: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{session_id}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **data) -> None:
        record = {"ts": now_ms(), "event": event, **data}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class MediaRecorder:
    def __init__(self, root: Path):
        self.video_dir = root / "video"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._audio_path = root / "audio.wav"
        self._wav: sf.SoundFile | None = None

    def add_frame(self, ts: int, jpeg: bytes) -> None:
        (self.video_dir / f"{ts}.jpg").write_bytes(jpeg)

    def add_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._wav is None:
            self._wav = sf.SoundFile(self._audio_path, "w", samplerate=sample_rate, channels=1, subtype="PCM_16")
        self._wav.write(samples)

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
