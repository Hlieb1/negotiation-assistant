import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.video.face_emotion import EMOTIONS, FaceEmotionRecognizer


def evaluate(recording: Path, detector: str, classifier: str, hsemotion_model: str, skip_s: float) -> pd.DataFrame:
    frames = pd.read_csv(recording / "frames.csv")
    frames = frames[frames.expected.notna() & (frames.segment_t >= skip_s)]
    recognizer = FaceEmotionRecognizer(detector, classifier, hsemotion_model)
    recognizer.warm_up()
    rows = []
    for row in frames.itertuples():
        result = recognizer.analyze(cv2.imread(str(recording / f"{row.n:04d}.jpg")), row.ts)
        predicted = EMOTIONS[int(result.probs.argmax())] if result.face_found else "no_face"
        rows.append({"expected": row.expected, "predicted": predicted, "processing_ms": result.processing_ms})
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Точність класифікаторів емоцій на записі tools/face_demo.py --scenario --record")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--detector", default="opencv")
    parser.add_argument("--classifiers", nargs="+", default=["deepface", "hsemotion"])
    parser.add_argument("--hsemotion-model", default="enet_b2_7")
    parser.add_argument("--skip-s", type=float, default=1.5, help="початок кожного відрізка, коли вираз ще змінюється")
    args = parser.parse_args()

    summary = []
    for classifier in args.classifiers:
        result = evaluate(args.recording, args.detector, classifier, args.hsemotion_model, args.skip_s)
        result.to_csv(args.recording / f"eval_{classifier}.csv", index=False)
        print(f"\n{classifier}: accuracy {(result.expected == result.predicted).mean():.2f}, "
              f"median {result.processing_ms.median():.0f} ms")
        print(pd.crosstab(result.expected, result.predicted))
        summary.append({
            "classifier": classifier,
            "accuracy": round((result.expected == result.predicted).mean(), 3),
            "median_ms": round(result.processing_ms.median(), 1),
            **{f"acc_{e}": round((g.predicted == e).mean(), 3) for e, g in result.groupby("expected")},
        })
    pd.DataFrame(summary).to_csv(args.recording / "eval_summary.csv", index=False)
    print(f"\nЗбережено: {args.recording / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
