import base64
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import SERVER_DIR, Settings
from app.decision import build_decisions
from app.main import create_app
from app.pipeline import Models
from app.video.face_emotion import EMOTIONS, FaceResult

REPLIES = json.loads((SERVER_DIR / "app/decision/replies_uk.json").read_text("utf-8"))["replies"]


class FakeRecognizer:
    def analyze(self, frame, ts):
        probs = np.zeros(len(EMOTIONS))
        probs[EMOTIONS.index("happy")] = 1.0
        return FaceResult(ts=ts, probs=probs, box=(0, 0, 100, 100))


def make_client(tmp_path) -> TestClient:
    settings = Settings.model_validate({
        "decision": {"min_interval_s": 0, "stable_s": 0, "llm": {"enabled": False}},
        "storage": {"log_dir": str(tmp_path / "logs"), "record_media": True,
                    "recordings_dir": str(tmp_path / "rec")},
    })
    models = Models(face=FakeRecognizer(), executor=ThreadPoolExecutor(max_workers=1),
                    decisions=build_decisions(settings, warm_up_llm=False))
    return TestClient(create_app(settings, models))


def jpeg_b64() -> str:
    _, jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    return base64.b64encode(jpeg.tobytes()).decode()


def test_session_round_trip(tmp_path):
    with make_client(tmp_path).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "session", "action": "start", "mode": "fuzzy"}))
        status = ws.receive_json()
        assert status["session"] == "started" and status["mode"] == "fuzzy"

        ws.send_text(json.dumps({"type": "video_frame", "ts": 1, "jpeg_b64": jpeg_b64()}))
        hint = ws.receive_json()
        assert hint["type"] == "hint" and hint["mode"] == "fuzzy" and hint["latency_ms"] >= 0
        assert hint["state"] == "згода" and hint["confidence"] > 0.3
        assert hint["reply"] in REPLIES["close"] and hint["tone"] == "впевнений, чіткий"

        pcm = base64.b64encode(np.zeros(32000, dtype="<i2").tobytes()).decode()
        ws.send_text(json.dumps({"type": "audio_chunk", "ts": 2, "pcm16_b64": pcm}))

        ws.send_text(json.dumps({"type": "session", "action": "stop"}))
        assert ws.receive_json()["session"] == "stopped"

    lines = [json.loads(line) for line in next((tmp_path / "logs").glob("*.jsonl")).read_text("utf-8").splitlines()]
    assert lines[0]["event"] == "session_start"
    hints = [line for line in lines if line["event"] == "hint"]
    assert len(hints) == 1 and hints[0]["source"] == "fuzzy" and hints[0]["strategy"] == "close"
    assert lines[-1]["event"] == "session_stop"
    assert lines[-1]["frames"] == 1 and lines[-1]["audio_seconds"] == 2.0
    assert len(list((tmp_path / "rec").rglob("*.jpg"))) == 1
    assert len(list((tmp_path / "rec").rglob("audio.wav"))) == 1


def test_invalid_payload_returns_error(tmp_path):
    with make_client(tmp_path).websocket_connect("/ws") as ws:
        ws.send_text("{}")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "session", "action": "start"}))
        ws.receive_json()
        ws.send_text(json.dumps({"type": "video_frame", "ts": 1, "jpeg_b64": "bm90IGEganBlZw=="}))
        error = ws.receive_json()
        assert error["type"] == "error" and "JPEG" in error["detail"]
