import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Settings
from .protocol import Hint, Mode, now_ms
from .video.face_emotion import EMOTIONS, EMOTIONS_UK, FaceEmotionRecognizer
from .video.preprocessing import FrameSampler, prepare_frame
from .video.smoothing import EmotionSmoother


@dataclass
class Models:
    face: FaceEmotionRecognizer | None
    executor: ThreadPoolExecutor


def load_models(settings: Settings) -> Models:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="models")
    face = None
    if settings.video.enabled:
        video = settings.video
        face = FaceEmotionRecognizer(video.detector_backend, video.align, video.min_face_size)
        executor.submit(face.warm_up).result()
    return Models(face=face, executor=executor)


class Pipeline:
    def __init__(self, mode: Mode, settings: Settings, models: Models, log: Callable[..., None]):
        self.mode = mode
        self.settings = settings
        self.models = models
        self.log = log
        self.min_interval_ms = int(settings.session.hint_min_interval_s * 1000)
        self.sampler = FrameSampler(settings.video.max_fps)
        self.smoother = EmotionSmoother(settings.video.smoothing_window_s, settings.video.min_face_ratio)
        self.frames = 0
        self.frames_analyzed = 0
        self.audio_seconds = 0.0
        self._pending_frame: tuple[np.ndarray, int, int] | None = None
        self._frame_ready = asyncio.Event()
        self._last_data_ms: int | None = None
        self._last_hint_ms = 0
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if self.models.face is not None:
            self._tasks.append(asyncio.create_task(self._video_loop()))

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
        self._last_data_ms = recv_ms

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
            self.smoother.add(result)
            self._last_data_ms = recv_ms
            self.log(
                "face",
                frame_ts=ts,
                face_found=result.face_found,
                probs=dict(zip(EMOTIONS, np.round(result.probs, 4).tolist())) if result.face_found else None,
                box=result.box,
                processing_ms=round(result.processing_ms, 1),
            )

    def poll(self) -> Hint | None:
        now = now_ms()
        if self._last_data_ms is None or now - self._last_hint_ms < self.min_interval_ms:
            return None
        face = self.smoother.current()
        if face is None:
            state, confidence, reply = "обличчя не знайдено", 0.0, "Направте камеру на обличчя співрозмовника."
        else:
            top = np.argsort(face.probs)[::-1][:3]
            state = EMOTIONS_UK[face.dominant]
            confidence = float(face.probs.max())
            reply = ", ".join(f"{EMOTIONS_UK[EMOTIONS[i]]} {face.probs[i]:.2f}" for i in top)
        hint = Hint(ts=now, state=state, confidence=confidence, reply=reply, tone="—",
                    mode=self.mode, latency_ms=now - self._last_data_ms)
        self._last_hint_ms = now
        self._last_data_ms = None
        return hint
