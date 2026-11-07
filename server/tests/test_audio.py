import numpy as np
import pytest

from app.audio.asr import UtteranceBuffer
from app.audio.features import pause_stats, voice_features
from app.audio.preprocessing import SAMPLE_RATE, SpeechDetector, SpeechSegment
from app.audio.voice_emotion import to_emotion_vector
from app.video.face_emotion import EMOTIONS

FRAME = 480


def segment_from_mask(mask: list[int]) -> SpeechSegment:
    return SpeechSegment(audio=np.zeros(len(mask) * FRAME, dtype=np.float32), mask=np.array(mask, dtype=bool),
                         frame_len=FRAME)


def harmonic_voice(f0: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    return (0.2 * wave * envelope).astype(np.float32)


def test_silence_has_no_speech():
    segment = SpeechDetector(denoise=False).process(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    assert not segment.has_speech and segment.speech_ratio == 0.0
    assert segment.trailing_silence_s() == pytest.approx(2.0)
    assert voice_features(segment)["speech_s"] == 0.0


def test_pause_stats_counts_inner_pauses_only():
    mask = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
    ratio, pauses = pause_stats(segment_from_mask(mask))
    assert pauses == 1
    assert ratio == pytest.approx(8 / 13)


def test_features_measure_pitch_of_voiced_signal():
    audio = harmonic_voice(150, 2.0)
    segment = SpeechSegment(audio=audio, mask=np.ones(len(audio) // FRAME, dtype=bool), frame_len=FRAME)
    features = voice_features(segment)
    assert features["f0_mean_hz"] == pytest.approx(150, rel=0.05)
    assert features["f0_std_st"] < 0.5
    assert len(features["mfcc"]) == 13


def test_speech_span_keeps_inner_silence():
    segment = segment_from_mask([0, 1, 0, 0, 1, 0])
    assert len(segment.speech_span()) == 4 * FRAME
    assert len(segment.speech_audio()) == 2 * FRAME
    assert segment.trailing_silence_s() == pytest.approx(FRAME / SAMPLE_RATE)


def test_emotion2vec_labels_map_to_seven_emotions():
    labels = ["生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy", "中立/neutral", "其他/other",
              "难过/sad", "吃惊/surprised", "<unk>"]
    scores = [0.1, 0.0, 0.0, 0.4, 0.2, 0.1, 0.1, 0.0, 0.1]
    probs, unknown = to_emotion_vector(labels, scores)
    assert unknown == pytest.approx(0.2)
    assert probs.sum() == pytest.approx(1.0)
    assert EMOTIONS[int(probs.argmax())] == "happy"


def test_utterance_buffer_flushes_on_pause_and_length():
    buffer = UtteranceBuffer(min_pause_s=0.6, max_utterance_s=3.0)
    speech = np.ones(SAMPLE_RATE, dtype=np.float32)
    assert buffer.add(speech, trailing_silence_s=0.1, ts=1000) is None
    audio, ts = buffer.add(speech, trailing_silence_s=0.8, ts=3000)
    assert ts == 1000 and len(audio) == 2 * SAMPLE_RATE
    assert buffer.add(np.zeros(0, dtype=np.float32), trailing_silence_s=2.0, ts=5000) is None
    for i in range(2):
        assert buffer.add(speech, 0.0, 7000 + i) is None
    assert len(buffer.add(speech, 0.0, 9000)[0]) == 3 * SAMPLE_RATE
