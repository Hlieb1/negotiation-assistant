import asyncio
import base64
from datetime import datetime
from typing import Awaitable, Callable

import cv2
import numpy as np
from pydantic import BaseModel

from .config import Settings, resolve_path
from .pipeline import Models, Pipeline
from .audio.preprocessing import SpeechDetector, pcm16_to_float
from .protocol import AudioChunk, Enrolled, Error, Mode, SessionControl, Status, VideoFrame, VoiceEnroll, now_ms
from .session_log import MediaRecorder, SessionLog

POLL_INTERVAL_S = 0.2


def decode_jpeg(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid JPEG")
    return image


def decode_pcm16(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise ValueError("PCM16 payload has odd length")
    return np.frombuffer(data, dtype="<i2")


class Session:
    def __init__(self, settings: Settings, models: Models, send: Callable[[str], Awaitable[None]]):
        self.settings = settings
        self.models = models
        self._send_text = send
        self.session_id: str | None = None
        self.mode: Mode = settings.session.default_mode
        self.pipeline: Pipeline | None = None
        self.log: SessionLog | None = None
        self.recorder: MediaRecorder | None = None
        self._hint_task: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        return self.pipeline is not None

    async def send(self, message: BaseModel) -> None:
        await self._send_text(message.model_dump_json())

    async def handle(self, message: VideoFrame | AudioChunk | SessionControl | VoiceEnroll, recv_ms: int) -> None:
        if isinstance(message, VoiceEnroll):
            await self.enroll(message)
            return
        if isinstance(message, SessionControl):
            if message.action == "start":
                await self.start(message.mode or self.settings.session.default_mode, message.context)
            else:
                await self.stop()
            return
        if not self.active:
            return
        try:
            if isinstance(message, VideoFrame):
                jpeg = base64.b64decode(message.jpeg_b64, validate=True)
                self.pipeline.add_video(decode_jpeg(jpeg), message.ts, recv_ms)
                if self.recorder:
                    self.recorder.add_frame(message.ts, jpeg)
            else:
                samples = decode_pcm16(base64.b64decode(message.pcm16_b64, validate=True))
                self.pipeline.add_audio(samples, message.sample_rate, message.ts, recv_ms)
                if self.recorder:
                    self.recorder.add_audio(samples, message.sample_rate)
        except ValueError as exc:
            await self.send(Error(ts=now_ms(), detail=f"{message.type}: {exc}"))

    async def enroll(self, message: VoiceEnroll) -> None:
        verifier = self.models.speaker
        if verifier is None:
            await self.send(Enrolled(ts=now_ms(), ok=False, speech_s=0, detail="розпізнавання мовця вимкнено"))
            return
        try:
            samples = decode_pcm16(base64.b64decode(message.pcm16_b64, validate=True))
        except ValueError as exc:
            await self.send(Enrolled(ts=now_ms(), ok=False, speech_s=0, detail=str(exc)))
            return
        audio = self.settings.audio
        detector = SpeechDetector(audio.vad_aggressiveness, audio.vad_frame_ms, audio.min_rms_db,
                                  target_level_db=audio.target_level_db, max_gain_db=audio.max_gain_db,
                                  noise_margin_db=audio.noise_margin_db)
        speech = detector.process(pcm16_to_float(samples, message.sample_rate)).speech_audio()
        speech_s = round(len(speech) / 16000, 1)
        if speech_s < self.settings.speaker.min_enroll_s:
            await self.send(Enrolled(ts=now_ms(), ok=False, speech_s=speech_s,
                                     detail=f"замало мови: {speech_s} с, потрібно щонайменше "
                                            f"{self.settings.speaker.min_enroll_s:g} с"))
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.models.audio_executor, verifier.enroll, speech)
        await self.send(Enrolled(ts=now_ms(), ok=True, speech_s=speech_s, detail="зразок голосу збережено"))

    async def start(self, mode: Mode, context: str = "") -> None:
        if self.active:
            await self.stop()
        storage = self.settings.storage
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mode = mode
        self.log = SessionLog(resolve_path(storage.log_dir), self.session_id)
        voice_profile = self.models.speaker is not None and self.models.speaker.enrolled
        self.log.write("session_start", mode=mode, context=context, voice_profile=voice_profile,
                       config=self.settings.model_dump(mode="json"))
        self.pipeline = Pipeline(mode, self.settings, self.models, self.log.write, context)
        self.pipeline.start()
        if storage.record_media:
            self.recorder = MediaRecorder(resolve_path(storage.recordings_dir) / self.session_id)
        self._hint_task = asyncio.create_task(self._hint_loop())
        await self.send(Status(ts=now_ms(), session="started", mode=mode, session_id=self.session_id,
                               voice_profile=voice_profile))

    async def stop(self) -> None:
        if not self.active:
            return
        self._hint_task.cancel()
        try:
            await self._hint_task
        except asyncio.CancelledError:
            pass
        await self.pipeline.stop()
        self.log.write("session_stop", frames=self.pipeline.frames, frames_analyzed=self.pipeline.frames_analyzed,
                       audio_seconds=round(self.pipeline.audio_seconds, 2))
        self.log.close()
        if self.recorder:
            self.recorder.close()
        session_id = self.session_id
        self.pipeline = self.log = self.recorder = self._hint_task = None
        try:
            await self.send(Status(ts=now_ms(), session="stopped", mode=self.mode, session_id=session_id))
        except Exception:
            pass

    async def _hint_loop(self) -> None:
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            hint = await self.pipeline.next_hint()
            if hint is not None:
                await self.send(hint)
