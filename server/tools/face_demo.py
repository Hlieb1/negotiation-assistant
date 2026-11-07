import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.config import load_settings
from app.video.face_emotion import EMOTIONS, EMOTIONS_UK, FaceEmotionRecognizer
from app.video.preprocessing import FrameSampler, prepare_frame
from app.video.smoothing import EmotionSmoother

SCENARIO = [
    ("neutral", "Нейтральне обличчя"),
    ("happy", "Посміхніться"),
    ("angry", "Насуптеся, покажіть злість"),
    ("surprise", "Здивуйтеся"),
    ("sad", "Покажіть смуток"),
    ("neutral", "Знову нейтральне обличчя"),
]


def draw(frame, result, smoothed, fps, prompt):
    if result.box:
        x, y, w, h = result.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(frame, f"{fps:.1f} fps, {result.processing_ms:.0f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if prompt:
        cv2.putText(frame, prompt, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
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
    parser.add_argument("--classifier", choices=["hsemotion", "deepface"], help="перевизначає video.classifier")
    parser.add_argument("--scenario", action="store_true", help="по черзі просить показати емоції")
    parser.add_argument("--segment-s", type=float, default=6.0)
    parser.add_argument("--record", action="store_true", help="зберігати кадри і ймовірності в recordings/")
    args = parser.parse_args()

    video = load_settings().video
    detector = args.detector or video.detector_backend
    classifier = args.classifier or video.classifier
    recognizer = FaceEmotionRecognizer(detector, classifier, video.hsemotion_model,
                                       video.min_face_size, video.face_margin)
    recognizer.warm_up()
    sampler = FrameSampler(video.max_fps)
    smoother = EmotionSmoother(video.smoothing_window_s, video.min_face_ratio)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    writer = None
    if args.record:
        out_dir = SERVER_DIR / "recordings" / f"face_demo_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True)
        csv_file = (out_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["n", "ts", "expected", "segment_t", "face_found", "x", "y", "w", "h",
                         *EMOTIONS, "smoothed", "processing_ms", "detector", "classifier"])
        print(f"Запис у {out_dir}")
    print("Вихід: клавіша q або Esc.")

    started = time.monotonic() + (5 if args.scenario else 0)
    segment = -1
    result = None
    analyzed = []
    n = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            print("Камеру не знайдено.")
            break
        expected, prompt, segment_t = "", "", 0.0
        elapsed = time.monotonic() - started
        if args.scenario and elapsed < 0:
            prompt = f"GET READY {-elapsed:.0f}"
        elif args.scenario:
            index = int(elapsed // args.segment_s)
            if index >= len(SCENARIO):
                break
            expected = SCENARIO[index][0]
            segment_t = elapsed - index * args.segment_s
            prompt = f"{expected.upper()}  {args.segment_s - segment_t:.0f}s"
            if index != segment:
                segment = index
                print(f">>> {SCENARIO[index][1]}")
        ts = time.time_ns() // 1_000_000
        if sampler.accept(ts):
            raw = frame.copy()
            result = recognizer.analyze(prepare_frame(frame, video.max_width, video.equalize), ts)
            smoother.add(result)
            smoothed = smoother.current()
            analyzed = [t for t in analyzed if ts - t < 2000] + [ts]
            label = EMOTIONS_UK[smoothed.dominant] if smoothed else "обличчя не знайдено"
            print(f"{label:20s} {result.processing_ms:6.0f} мс")
            if writer and (expected or not args.scenario):
                cv2.imwrite(str(out_dir / f"{n:04d}.jpg"), raw)
                probs = [f"{p:.4f}" for p in result.probs] if result.face_found else [""] * len(EMOTIONS)
                writer.writerow([n, ts, expected, f"{segment_t:.2f}", int(result.face_found),
                                 *(result.box or ("", "", "", "")), *probs,
                                 smoothed.dominant if smoothed else "", f"{result.processing_ms:.1f}", detector, classifier])
                n += 1
        if result is not None:
            draw(frame, result, smoother.current(), len(analyzed) / 2, prompt)
        cv2.imshow("face emotion", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    capture.release()
    cv2.destroyAllWindows()
    if writer:
        csv_file.close()
        print(f"Збережено кадрів: {n}")


if __name__ == "__main__":
    main()
