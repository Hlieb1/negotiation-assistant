import numpy as np
import pytest

from app.video.face_emotion import EMOTIONS, FaceResult, to_emotion_vector
from app.video.preprocessing import FrameSampler, prepare_frame
from app.video.smoothing import EmotionSmoother


def one_hot(emotion: str) -> np.ndarray:
    probs = np.zeros(len(EMOTIONS))
    probs[EMOTIONS.index(emotion)] = 1.0
    return probs


def test_sampler_limits_rate():
    sampler = FrameSampler(max_fps=4)
    accepted = [ts for ts in range(0, 2000, 100) if sampler.accept(ts)]
    assert len(accepted) == 8


def test_sampler_tolerates_jitter_at_target_rate():
    sampler = FrameSampler(max_fps=4)
    timestamps = [0, 245, 505, 740, 1010, 1240, 1500, 1755]
    assert all(sampler.accept(ts) for ts in timestamps)


def test_sampler_recovers_after_clock_jump():
    sampler = FrameSampler(max_fps=4)
    assert sampler.accept(100_000)
    assert sampler.accept(1_000)
    assert not sampler.accept(1_100)


def test_prepare_frame_downscales_keeping_aspect():
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    assert prepare_frame(frame, max_width=640, equalize=True).shape == (480, 640, 3)
    assert prepare_frame(frame[:240, :320], max_width=640, equalize=False).shape == (240, 320, 3)


def test_smoother_averages_within_window():
    smoother = EmotionSmoother(window_s=1.0, min_face_ratio=0.3)
    smoother.add(FaceResult(ts=0, probs=one_hot("angry")))
    smoother.add(FaceResult(ts=500, probs=one_hot("happy")))
    smoother.add(FaceResult(ts=900, probs=one_hot("happy")))
    smoothed = smoother.current()
    assert smoothed.dominant == "happy" and smoothed.samples == 3
    smoother.add(FaceResult(ts=1400, probs=one_hot("happy")))
    assert smoother.current().samples == 3
    assert smoother.current().probs[EMOTIONS.index("angry")] == 0


def test_smoother_reports_missing_face():
    smoother = EmotionSmoother(window_s=1.0, min_face_ratio=0.5)
    assert smoother.current() is None
    smoother.add(FaceResult(ts=0, probs=one_hot("sad")))
    for ts in (200, 400, 600):
        smoother.add(FaceResult(ts=ts, probs=None))
    assert smoother.current() is None
    smoother.add(FaceResult(ts=800, probs=one_hot("sad")))
    assert smoother.current() is None
    smoother.add(FaceResult(ts=1000, probs=one_hot("sad")))
    assert smoother.current().face_ratio == 0.5


def test_hsemotion_eight_classes_map_to_seven():
    labels = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Neutral", 6: "Sadness",
              7: "Surprise"}
    probs = to_emotion_vector(labels, np.array([0.1, 0.2, 0.1, 0.0, 0.3, 0.2, 0.1, 0.0]))
    assert probs[EMOTIONS.index("disgust")] == pytest.approx(0.3)
    assert probs[EMOTIONS.index("happy")] == pytest.approx(0.3)
    assert probs.sum() == pytest.approx(1.0)
