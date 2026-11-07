import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio.asr import SpeechRecognizer, Transcript, UtteranceBuffer
from .audio.features import voice_features
from .audio.preprocessing import SpeechDetector, pcm16_to_float
from .audio.voice_emotion import VoiceEmotionClassifier, VoiceResult
from .config import Settings
from .protocol import Hint, Mode, now_ms
from .video.face_emotion import EMOTIONS, EMOTIONS_UK, FaceEmotionRecognizer
from .video.preprocessing import FrameSampler, prepare_frame
from .video.smoothing import EmotionSmoother

logger = logging.getLogger("negotiation")
AUDIO_QUEUE_SIZE = 3


@dataclass
class Models:
    face: FaceEmotionRecognizer | None
    executor: ThreadPoolExecutor
    voice: VoiceEmotionClassifier | None = None
    asr: SpeechRecognizer | None = None
    audio_executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "audio"))
    asr_executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, "asr"))

    def shutdown(self) -> None:
        for executor in (self.executor, self.audio_executor, self.asr_executor):
            executor.shutdown(wait=False, cancel_futures=True)


def load_models(settings: Settings) -> Models:
    models = Models(face=None, executor=ThreadPoolExecutor(1, "video"))
    video, audio, asr = settings.video, settings.audio, settings.asr
    if video.enabled:
        models.face = FaceEmotionRecognizer(video.detector_backend, video.classifier, video.hsemotion_model,
                                            video.min_face_size, video.face_margin)
        models.executor.submit(models.face.warm_up).result()
    if audio.enabled:
        models.voice = VoiceEmotionClassifier(audio.voice_model, audio.voice_device, audio.min_speech_s)
        models.audio_executor.submit(models.voice.warm_up).result()
    if asr.enabled:
        models.asr = SpeechRecognizer(asr.model, asr.language, asr.device, asr.compute_type, asr.beam_size)
        models.asr_executor.submit(models.asr.warm_up).result()
    return models


class Pipeline:
    def __init__(self, mode: Mode, settings: Settings, models: Models, log: Callable[..., None]):
        self.mode = mode
        self.settings = settings
        self.models = models
        self.log = log
        self.min_interval_ms = int(settings.session.hint_min_interval_s * 1000)
        self.sampler = FrameSampler(settings.video.max_fps)
        self.face_smoother = EmotionSmoother(settings.video.smoothing_window_s, settings.video.min_face_ratio)
        self.voice_smoother = EmotionSmoother(settings.audio.smoothing_window_s)
        audio = settings.audio
        self.speech_detector = SpeechDetector(audio.vad_aggressiveness, audio.vad_frame_ms, audio.min_rms_db,
                                              denoise=audio.denoise)
        self.utterances = UtteranceBuffer(settings.asr.min_pause_s, settings.asr.max_utterance_s)
        self.frames = 0
        self.frames_analyzed = 0
        self.audio_seconds = 0.0
        self.voice_features: dict = {}
        self.transcripts: list[Transcript] = []
        self._pending_frame: tuple[np.ndarray, int, int] | None = None
        self._frame_ready = asyncio.Event()
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_SIZE)
        self._last_data_ms: int | None = None
        self._last_hint_ms = 0
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if self.models.face is not None:
            self._tasks.append(asyncio.create_task(self._video_loop()))
        if self.settings.audio.enabled:
            self._tasks.append(asyncio.create_task(self._audio_loop()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def add_video(self, frame: np.ndarray, ts: int, recv_ms: int) -> None:
        self.frames += 1
        if self.models.face is not None and self.sampler.accept(ts):
            self._pending_frame = (frame, ts, recv_ms)
            self._frame_ready.set()

    def add_audio(self, samples: np.ndarray, sample_rate: int, ts: int, recv_ms: int) -> None:
        self.audio_seconds += len(samples) / sample_rate
        if not self.settings.audio.enabled:
            return
        if self._audio_queue.full():
            self._audio_queue.get_nowait()
            logger.warning("audio processing is behind real time, dropping a chunk")
        self._audio_queue.put_nowait((samples, sample_rate, ts, recv_ms))

    async def _video_loop(self) -> None:
        loop = asyncio.get_running_loop()
        video = self.settings.video
        while True:
            await self._frame_ready.wait()
            self._frame_ready.clear()
            frame, ts, recv_ms = self._pending_frame
            prepared = prepare_frame(frame, video.max_width, video.equalize)
            result = await loop.run_in_executor(self.models.executor, self.models.face.analyze, prepared, ts)
            self.frames_analyzed += 1
            self.face_smoother.add(result)
            self._last_data_ms = recv_ms
            self.log(
                "face",
                frame_ts=ts,
                face_found=result.face_found,
                probs=dict(zip(EMOTIONS, np.round(result.probs, 4).tolist())) if result.face_found else None,
                box=result.box,
                processing_ms=round(result.processing_ms, 1),
            )

    def _analyze_audio(self, samples: np.ndarray, sample_rate: int, ts: int):
        segment = self.speech_detector.process(pcm16_to_float(samples, sample_rate))
        features = voice_features(segment)
        voice = self.models.voice.predict(segment.speech_span(), ts) if self.models.voice else VoiceResult(ts, None)
        return segment, features, voice

    async def _audio_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            samples, sample_rate, ts, recv_ms = await self._audio_queue.get()
            started = now_ms()
            segment, features, voice = await loop.run_in_executor(
                self.models.audio_executor, self._analyze_audio, samples, sample_rate, ts)
            self.voice_smoother.add(voice)
            if segment.has_speech:
                self.voice_features = features
            self._last_data_ms = recv_ms
            self.log(
                "voice",
                chunk_ts=ts,
                features={k: v for k, v in features.items() if k != "mfcc"},
                probs=dict(zip(EMOTIONS, np.round(voice.probs, 4).tolist())) if voice.probs is not None else None,
                unknown=round(voice.unknown, 4),
                emotion_ms=round(voice.processing_ms, 1),
                processing_ms=now_ms() - started,
            )
            if self.models.asr is not None:
                utterance = self.utterances.add(segment.speech_span(), segment.trailing_silence_s(), ts)
                if utterance is not None:
                    task = asyncio.create_task(self._transcribe(*utterance, recv_ms))
                    self._tasks.append(task)
                    task.add_done_callback(lambda t: t in self._tasks and self._tasks.remove(t))

    async def _transcribe(self, audio: np.ndarray, ts: int, recv_ms: int) -> None:
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(self.models.asr_executor, self.models.asr.transcribe, audio, ts)
        if transcript.text:
            self.transcripts.append(transcript)
            self._last_data_ms = recv_ms
        self.log("transcript", utterance_ts=ts, text=transcript.text, duration_s=transcript.duration_s,
                 processing_ms=round(transcript.processing_ms, 1))

    def poll(self) -> Hint | None:
        now = now_ms()
        if self._last_data_ms is None or now - self._last_hint_ms < self.min_interval_ms:
            return None
        face = self.face_smoother.current()
        voice = self.voice_smoother.current()
        parts = []
        if face is not None:
            parts.append(f"обличчя: {EMOTIONS_UK[face.dominant]} {face.probs.max():.2f}")
        if voice is not None:
            parts.append(f"голос: {EMOTIONS_UK[voice.dominant]} {voice.probs.max():.2f}")
        if self.transcripts:
            parts.append(f"«{self.transcripts[-1].text}»")
        main = face or voice
        hint = Hint(
            ts=now,
            state=EMOTIONS_UK[main.dominant] if main else "немає даних",
            confidence=float(main.probs.max()) if main else 0.0,
            reply="; ".join(parts) or "Направте камеру на співрозмовника і говоріть.",
            tone="—",
            mode=self.mode,
            latency_ms=now - self._last_data_ms,
        )
        self._last_hint_ms = now
        self._last_data_ms = None
        return hint
