import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient

from app.audio.asr import Transcript
from app.audio.speaker import ME, OTHER, SpeakerResult
from app.config import Settings
from app.decision import build_decisions
from app.decision.base import StateVector
from app.decision.llm import describe_situation
from app.fusion.fusion import FusedEmotion
from app.fusion.states import NegotiationState, StateStabilizer
from app.hint import HintPolicy
from app.main import create_app
from app.pipeline import Models, Pipeline, Turn, format_dialogue
from app.video.face_emotion import EMOTIONS

STATES = ("злість", "напруга", "сумнів", "згода", "нейтральний")


def negotiation_state(degrees: dict) -> NegotiationState:
    fused = FusedEmotion(probs=np.full(len(EMOTIONS), 1 / 7), valence=0, arousal=0.5, hesitation=0,
                         modalities=["face"], coverage=1.0)
    full = {s: degrees.get(s, 0.0) for s in STATES}
    return NegotiationState(name=max(full, key=full.get), degrees=full, confidence=0.4, fused=fused)


def test_stabilizer_keeps_state_until_margin():
    stabilizer = StateStabilizer(margin=0.05)
    assert stabilizer.update(negotiation_state({"злість": 0.4, "нейтральний": 0.3})).name == "злість"
    assert stabilizer.update(negotiation_state({"злість": 0.36, "нейтральний": 0.39})).name == "злість"
    assert stabilizer.update(negotiation_state({"злість": 0.3, "нейтральний": 0.4})).name == "нейтральний"


def test_speech_trigger_respects_its_own_interval():
    policy = HintPolicy(min_interval_s=4, stable_s=1, min_confidence=0.1, speech_min_interval_s=2)
    state = StateVector("сумнів", {s: 0.2 for s in STATES}, 0.4)
    assert policy.should_hint_on_speech(state, 0)
    assert not policy.should_hint_on_speech(state, 1500)
    assert policy.should_hint_on_speech(state, 2500)
    assert not policy.should_hint(state, 9000)


def test_dialogue_and_context_reach_the_prompt():
    dialogue = format_dialogue([Turn(1, ME, "Ціна 60 тисяч."), Turn(2, OTHER, "Це задорого.")])
    assert dialogue == "Я: Ціна 60 тисяч.\nСпіврозмовник: Це задорого."
    situation = describe_situation(StateVector("злість", {"злість": 0.5}, 0.4), {}, dialogue, [], None, None,
                                   "Продаю CRM, знижка до 10 %.")
    assert "Продаю CRM" in situation and "Співрозмовник: Це задорого." in situation


class FakeASR:
    def transcribe(self, audio, ts):
        return Transcript(ts=ts, text="Це задорого.", duration_s=1.0, processing_ms=1.0)


class FakeSpeaker:
    enrolled = True

    def __init__(self, label):
        self.label = label

    def identify(self, audio):
        return SpeakerResult(self.label, 0.1 if self.label == OTHER else 0.7)


def make_pipeline(label: str, logged: list) -> Pipeline:
    settings = Settings.model_validate({"decision": {"llm": {"enabled": False}}})
    models = Models(face=None, executor=ThreadPoolExecutor(1), asr=FakeASR(), speaker=FakeSpeaker(label),
                    decisions=build_decisions(settings, warm_up_llm=False))
    return Pipeline("fuzzy", settings, models, lambda event, **data: logged.append((event, data)), "контекст")


def test_counterpart_phrase_is_pending_and_mine_is_not():
    logged = []
    other = make_pipeline(OTHER, logged)
    asyncio.run(other._transcribe(np.zeros(16000, dtype=np.float32), 1000, 1000))
    assert other._pending_phrase.text == "Це задорого." and other.dialogue[-1].speaker == OTHER
    mine = make_pipeline(ME, logged)
    asyncio.run(mine._transcribe(np.zeros(16000, dtype=np.float32), 1000, 1000))
    assert mine._pending_phrase is None and mine.dialogue[-1].speaker == ME
    assert [e for e, _ in logged] == ["transcript", "transcript"]


def test_enroll_without_speaker_model_reports_error(tmp_path):
    settings = Settings.model_validate({"storage": {"log_dir": str(tmp_path)}})
    models = Models(face=None, executor=ThreadPoolExecutor(1))
    with TestClient(create_app(settings, models)).websocket_connect("/ws") as ws:
        pcm = base64.b64encode(np.zeros(16000, dtype="<i2").tobytes()).decode()
        ws.send_text(json.dumps({"type": "enroll", "ts": 1, "pcm16_b64": pcm}))
        reply = ws.receive_json()
        assert reply["type"] == "enrolled" and reply["ok"] is False
