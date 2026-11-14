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


class SpeechDetector:
    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30, min_rms_db: float = -45.0,
                 hangover_frames: int = 3, denoise: bool = False):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_len = SAMPLE_RATE * frame_ms // 1000
        self.min_rms_db = min_rms_db
        self.hangover_frames = hangover_frames
        self.denoise = denoise

    def process(self, audio: np.ndarray) -> SpeechSegment:
        if self.denoise and len(audio) >= SAMPLE_RATE // 2:
            import noisereduce

            audio = noisereduce.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.8)
        audio = audio.astype(np.float32)
        n_frames = len(audio) // self.frame_len
        frames = audio[:n_frames * self.frame_len].reshape(n_frames, self.frame_len)
        rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
        pcm = (np.clip(frames, -1, 1) * 32767).astype("<i2")
        raw = np.array([self.vad.is_speech(f.tobytes(), SAMPLE_RATE) for f in pcm], dtype=bool)
        raw &= rms_db > self.min_rms_db
        return SpeechSegment(audio=audio, mask=self._hangover(raw), frame_len=self.frame_len)

    def _hangover(self, mask: np.ndarray) -> np.ndarray:
        smoothed = mask.copy()
        for i in np.flatnonzero(mask):
            smoothed[i:i + self.hangover_frames + 1] = True
        return smoothed
