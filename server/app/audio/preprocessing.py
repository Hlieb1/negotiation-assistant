from collections import deque
from dataclasses import dataclass

import librosa
import numpy as np
import webrtcvad

SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    audio: np.ndarray
    mask: np.ndarray
    frame_len: int
    gain_db: float = 0.0

    @property
    def speech_ratio(self) -> float:
        return float(self.mask.mean()) if len(self.mask) else 0.0

    @property
    def has_speech(self) -> bool:
        return bool(self.mask.any())

    def speech_audio(self) -> np.ndarray:
        if not self.has_speech:
            return np.zeros(0, dtype=np.float32)
        frames = np.flatnonzero(self.mask)
        return np.concatenate([self.audio[i * self.frame_len:(i + 1) * self.frame_len] for i in frames])

    def speech_span(self, padding_s: float = 0.0) -> np.ndarray:
        if not self.has_speech:
            return np.zeros(0, dtype=np.float32)
        frames = np.flatnonzero(self.mask)
        padding = int(padding_s * SAMPLE_RATE)
        start = max(0, frames[0] * self.frame_len - padding)
        end = min(len(self.audio), (frames[-1] + 1) * self.frame_len + padding)
        return self.audio[start:end]

    def trailing_silence_s(self) -> float:
        if not self.has_speech:
            return len(self.audio) / SAMPLE_RATE
        last = np.flatnonzero(self.mask)[-1]
        return (len(self.mask) - last - 1) * self.frame_len / SAMPLE_RATE


def pcm16_to_float(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = samples.astype(np.float32) / 32768.0
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio


def frame_levels_db(audio: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = len(audio) // frame_len
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)


class LevelNormalizer:
    def __init__(self, target_db: float = -20.0, max_gain_db: float = 30.0, history: int = 15):
        self.target_db = target_db
        self.max_gain_db = max_gain_db
        self._peaks: deque[float] = deque(maxlen=history)

    def gain_db(self, levels_db: np.ndarray) -> float:
        if len(levels_db):
            self._peaks.append(float(np.percentile(levels_db, 95)))
        if not self._peaks:
            return 0.0
        return float(np.clip(self.target_db - np.median(self._peaks), 0.0, self.max_gain_db))


class SpeechDetector:
    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30, min_rms_db: float = -55.0,
                 hangover_frames: int = 3, denoise: bool = False, target_level_db: float | None = -20.0,
                 max_gain_db: float = 30.0, noise_margin_db: float = 6.0):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_len = SAMPLE_RATE * frame_ms // 1000
        self.min_rms_db = min_rms_db
        self.hangover_frames = hangover_frames
        self.denoise = denoise
        self.normalizer = LevelNormalizer(target_level_db, max_gain_db) if target_level_db is not None else None
        self.noise_margin_db = noise_margin_db
        self._noise: deque[float] = deque(maxlen=15)

    def process(self, audio: np.ndarray) -> SpeechSegment:
        if self.denoise and len(audio) >= SAMPLE_RATE // 2:
            import noisereduce

            audio = noisereduce.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)
        audio = audio.astype(np.float32)
        gain_db = self.normalizer.gain_db(frame_levels_db(audio, self.frame_len)) if self.normalizer else 0.0
        if gain_db:
            audio = np.clip(audio * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)
        rms_db = frame_levels_db(audio, self.frame_len)
        if len(rms_db):
            self._noise.append(float(np.percentile(rms_db, 10)))
        threshold = max(self.min_rms_db, float(np.median(self._noise)) + self.noise_margin_db) if self._noise             else self.min_rms_db
        frames = audio[:len(rms_db) * self.frame_len].reshape(len(rms_db), self.frame_len)
        pcm = (np.clip(frames, -1, 1) * 32767).astype("<i2")
        raw = np.array([self.vad.is_speech(f.tobytes(), SAMPLE_RATE) for f in pcm], dtype=bool)
        raw &= rms_db > threshold
        return SpeechSegment(audio=audio, mask=self._hangover(raw), frame_len=self.frame_len, gain_db=gain_db)

    def _hangover(self, mask: np.ndarray) -> np.ndarray:
        smoothed = mask.copy()
        for i in np.flatnonzero(mask):
            smoothed[i:i + self.hangover_frames + 1] = True
        return smoothed
