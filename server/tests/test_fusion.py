import numpy as np
import pytest

from app.config import SERVER_DIR, Settings
from app.fusion.fusion import FusedEmotion, LateFusion, ProsodyBaseline
from app.fusion.states import StateMapper, load_rules, signals
from app.fusion.sync import synchronize
from app.video.face_emotion import EMOTIONS
from app.video.smoothing import SmoothedEmotion

RULES = load_rules(SERVER_DIR / Settings().fusion.rules_path)


def emotion(name: str, ts: int = 1000, strength: float = 1.0) -> SmoothedEmotion:
    probs = np.full(len(EMOTIONS), (1 - strength) / (len(EMOTIONS) - 1))
    probs[EMOTIONS.index(name)] = strength
    return SmoothedEmotion(ts=ts, probs=probs, found_ratio=1.0, samples=1)


def fusion() -> LateFusion:
    return LateFusion({"face": 0.65, "voice": 0.35}, {"face": 0.4, "voice": 0.6}, RULES["emotion_dimensions"])


def voice_features(f0=150.0, loudness=-30.0, f0_std=2.0, rate=5.0, pauses=0.2) -> dict:
    return {"f0_mean_hz": f0, "loudness_db": loudness, "f0_std_st": f0_std, "speech_rate": rate, "pause_ratio": pauses}


def test_sync_drops_stale_modality():
    observation = synchronize(emotion("happy", ts=10_000), emotion("angry", ts=2_000), {}, None, "", max_age_s=6)
    assert observation.modalities == ["face"] and observation.ts == 10_000
    assert synchronize(None, None, {}, None, "", 6) is None


def test_fusion_weights_and_renormalization():
    observation = synchronize(emotion("happy"), emotion("angry"), {}, None, "", 6)
    fused = fusion().fuse(observation, {})
    assert fused.probs[EMOTIONS.index("happy")] == pytest.approx(0.65)
    assert fused.probs[EMOTIONS.index("angry")] == pytest.approx(0.35)
    assert fused.valence == pytest.approx(0.4 * 0.8 + 0.6 * -0.8)
    face_only = fusion().fuse(synchronize(emotion("happy"), None, {}, None, "", 6), {})
    assert face_only.probs[EMOTIONS.index("happy")] == pytest.approx(1.0)
    assert face_only.coverage == pytest.approx(0.65)


def test_prosody_raises_arousal_and_hesitation():
    baseline = ProsodyBaseline(min_samples=3)
    assert baseline.zscores(voice_features()) == {}
    for f0 in (145, 150, 155):
        baseline.update(voice_features(f0=f0))
    calm = baseline.zscores(voice_features())
    excited = baseline.zscores(voice_features(f0=190, loudness=-22, f0_std=4, rate=7, pauses=0.6))
    assert all(abs(v) < 0.5 for v in calm.values())
    assert excited["loudness_db"] == 3 and excited["pitch_st"] > 2
    observation = synchronize(emotion("neutral"), None, excited, 1000, "", 6)
    fused = fusion().fuse(observation, excited)
    neutral = fusion().fuse(observation, calm)
    assert fused.arousal > neutral.arousal + 0.2
    assert fused.hesitation > 0.9 and neutral.hesitation < 0.1


@pytest.mark.parametrize("face, voice, expected", [
    ("angry", "angry", "злість"),
    ("fear", "fear", "напруга"),
    ("happy", "happy", "згода"),
    ("neutral", "neutral", "нейтральний"),
    ("sad", "sad", "сумнів"),
    ("disgust", "angry", "злість"),
])
def test_states_from_emotions(face, voice, expected):
    observation = synchronize(emotion(face), emotion(voice), {}, None, "", 6)
    state = StateMapper(RULES).map(fusion().fuse(observation, {}))
    assert state.name == expected
    assert sum(state.degrees.values()) == pytest.approx(1.0, abs=1e-3)


def test_hesitation_turns_neutral_face_into_doubt():
    fused = FusedEmotion(probs=emotion("neutral", strength=0.6).probs, valence=-0.05, arousal=0.4,
                         hesitation=1.0, modalities=["face", "voice"], coverage=1.0)
    assert StateMapper(RULES).map(fused).name == "сумнів"
    fused.hesitation = 0.0
    assert StateMapper(RULES).map(fused).name == "нейтральний"


def test_signals_are_bounded():
    fused = FusedEmotion(probs=emotion("angry").probs, valence=-0.8, arousal=0.95, hesitation=0.0,
                         modalities=["face"], coverage=0.65)
    values = signals(fused)
    assert all(0.0 <= v <= 1.0 for v in values.values())
    assert values["negativity"] == pytest.approx(0.8) and values["arousal_high"] == pytest.approx(0.9)


def test_rules_reject_unknown_terms(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("emotion_dimensions: {}\nstates:\n  x:\n    emotions: {rage: 1}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rage"):
        load_rules(path)
