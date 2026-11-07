import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings
from app.video.face_emotion import EMOTIONS, EMOTIONS_UK, FaceEmotionRecognizer
from app.video.preprocessing import FrameSampler, prepare_frame
from app.video.smoothing import EmotionSmoother


def draw(frame, result, smoothed, fps):
    if result.box:
        x, y, w, h = result.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    status = f"{fps:.1f} fps, {result.processing_ms:.0f} ms"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if smoothed is None:
        cv2.putText(frame, "no face", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    for i, (emotion, p) in enumerate(zip(EMOTIONS, smoothed.probs)):
        y = 55 + i * 24
        cv2.rectangle(frame, (110, y - 14), (110 + int(p * 200), y + 2), (0, 180, 255), -1)
        cv2.putText(frame, f"{emotion:9s}{p:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Розпізнавання емоцій за обличчям з вебкамери")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--detector", help="перевизначає video.detector_backend з config.yaml")
    args = parser.parse_args()

    video = load_settings().video
    recognizer = FaceEmotionRecognizer(args.detector or video.detector_backend, video.align, video.min_face_size)
    recognizer.warm_up()
    sampler = FrameSampler(video.max_fps)
    smoother = EmotionSmoother(video.smoothing_window_s, video.min_face_ratio)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    print("Вихід: клавіша q або Esc.")

    result = None
    analyzed = []
    while True:
        ok, frame = capture.read()
        if not ok:
            print("Камеру не знайдено.")
            break
        ts = time.time_ns() // 1_000_000
        if sampler.accept(ts):
            result = recognizer.analyze(prepare_frame(frame, video.max_width, video.equalize), ts)
            smoother.add(result)
            analyzed = [t for t in analyzed if ts - t < 2000] + [ts]
            smoothed = smoother.current()
            label = EMOTIONS_UK[smoothed.dominant] if smoothed else "обличчя не знайдено"
            print(f"{label:20s} {result.processing_ms:6.0f} мс")
        if result is not None:
            draw(frame, result, smoother.current(), len(analyzed) / 2)
        cv2.imshow("face emotion", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
