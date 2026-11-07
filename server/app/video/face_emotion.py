import time
from dataclasses import dataclass

import numpy as np

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EMOTIONS_UK = {
    "angry": "злість",
    "disgust": "відраза",
    "fear": "страх",
    "happy": "радість",
    "sad": "смуток",
    "surprise": "здивування",
    "neutral": "нейтральний",
}
HSEMOTION_LABELS = {
    "Anger": "angry",
    "Contempt": "disgust",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}


@dataclass
class FaceResult:
    ts: int
    probs: np.ndarray | None
    box: tuple[int, int, int, int] | None = None
    face_confidence: float = 0.0
    processing_ms: float = 0.0

    @property
    def face_found(self) -> bool:
        return self.probs is not None


def to_emotion_vector(labels: dict[int, str], scores: np.ndarray) -> np.ndarray:
    probs = np.zeros(len(EMOTIONS))
    for index, label in labels.items():
        probs[EMOTIONS.index(HSEMOTION_LABELS[label])] += scores[index]
    return probs / probs.sum()


class HSEmotionClassifier:
    def __init__(self, model_name: str = "enet_b2_7"):
        import urllib.request  # hsemotion_onnx uses it without importing
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self._model = HSEmotionRecognizer(model_name=model_name)

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        _, scores = self._model.predict_emotions(np.ascontiguousarray(face_bgr[:, :, ::-1]), logits=False)
        return to_emotion_vector(self._model.idx_to_class, scores)


class DeepFaceClassifier:
    def __init__(self):
        from deepface import DeepFace

        self._deepface = DeepFace

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        result = self._deepface.analyze(face_bgr, actions=["emotion"], detector_backend="skip",
                                        enforce_detection=False, silent=True)[0]["emotion"]
        probs = np.array([result[e] for e in EMOTIONS], dtype=np.float64)
        return probs / probs.sum()


def build_classifier(name: str, hsemotion_model: str):
    if name == "hsemotion":
        return HSEmotionClassifier(hsemotion_model)
    if name == "deepface":
        return DeepFaceClassifier()
    raise ValueError(f"unknown emotion classifier: {name}")


class FaceEmotionRecognizer:
    def __init__(self, detector_backend: str = "opencv", classifier: str = "hsemotion",
                 hsemotion_model: str = "enet_b2_7", min_face_size: int = 40, face_margin: float = 0.1):
        from deepface import DeepFace

        self._deepface = DeepFace
        self.detector_backend = detector_backend
        self.classifier = build_classifier(classifier, hsemotion_model)
        self.min_face_size = min_face_size
        self.face_margin = face_margin

    def warm_up(self) -> None:
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.classifier.predict(np.zeros((224, 224, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        try:
            faces = self._deepface.extract_faces(frame, detector_backend=self.detector_backend,
                                                 enforce_detection=True, align=False)
        except ValueError:
            return []
        boxes = [(f["facial_area"]["x"], f["facial_area"]["y"], f["facial_area"]["w"], f["facial_area"]["h"],
                  float(f.get("confidence") or 0.0)) for f in faces]
        return [b for b in boxes if min(b[2], b[3]) >= self.min_face_size]

    def analyze(self, frame: np.ndarray, ts: int) -> FaceResult:
        started = time.perf_counter()
        boxes = self.detect(frame)
        if not boxes:
            return FaceResult(ts=ts, probs=None, processing_ms=(time.perf_counter() - started) * 1000)
        x, y, w, h, confidence = max(boxes, key=lambda b: b[2] * b[3])
        margin = int(self.face_margin * w)
        crop = frame[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
        probs = self.classifier.predict(crop)
        return FaceResult(
            ts=ts,
            probs=probs,
            box=(x, y, w, h),
            face_confidence=confidence,
            processing_ms=(time.perf_counter() - started) * 1000,
        )
