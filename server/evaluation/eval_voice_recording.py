import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.audio.preprocessing import SpeechDetector
from app.audio.voice_emotion import VoiceEmotionClassifier
from app.video.face_emotion import EMOTIONS


def evaluate(recordings: list[Path], model: str, denoise: bool) -> pd.DataFrame:
    classifier = VoiceEmotionClassifier(model)
    classifier.warm_up()
    detector = SpeechDetector(denoise=denoise)
    rows = []
    for recording in recordings:
        for row in pd.read_csv(recording / "results.csv").itertuples():
            audio, _ = sf.read(recording / row.file, dtype="float32")
            segment = detector.process(audio)
            result = classifier.predict(segment.speech_span(), 0)
            predicted = EMOTIONS[int(result.probs.argmax())] if result.probs is not None else "no_speech"
            rows.append({"recording": recording.name, "file": row.file, "expected": row.expected,
                         "predicted": predicted, "processing_ms": result.processing_ms})
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Точність емоцій за голосом на записах tools/voice_demo.py --scenario")
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--models", nargs="+",
                        default=["emotion2vec/emotion2vec_plus_large", "emotion2vec/emotion2vec_plus_base"])
    parser.add_argument("--no-denoise", action="store_true")
    args = parser.parse_args()

    summary = []
    for model in args.models:
        result = evaluate(args.recordings, model, not args.no_denoise)
        name = model.split("/")[-1]
        result.to_csv(args.recordings[0] / f"eval_{name}.csv", index=False)
        accuracy = (result.expected == result.predicted).mean()
        print(f"\n{name}: accuracy {accuracy:.2f}, median {result.processing_ms.median():.0f} ms")
        print(pd.crosstab(result.expected, result.predicted))
        summary.append({
            "model": name,
            "samples": len(result),
            "accuracy": round(accuracy, 3),
            "median_ms": round(result.processing_ms.median(), 1),
            **{f"acc_{e}": round((g.predicted == e).mean(), 3) for e, g in result.groupby("expected")},
        })
    pd.DataFrame(summary).to_csv(args.recordings[0] / "eval_summary.csv", index=False)
    print(f"\nЗбережено: {args.recordings[0] / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
