import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.audio.preprocessing import SAMPLE_RATE, SpeechDetector
from app.audio.voice_emotion import VoiceEmotionClassifier
from app.video.face_emotion import EMOTIONS, EMOTIONS_UK

AUDIO_EXTENSIONS = {".wav", ".ogg", ".oga", ".opus", ".m4a", ".mp3", ".aac", ".flac", ".webm", ".amr", ".3gp"}
FOLDER_NAMES = {**{e: e for e in EMOTIONS}, **{uk: e for e, uk in EMOTIONS_UK.items()}}


def load_audio(path: Path) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def collect_samples(sources: list[Path]) -> list[tuple[str, Path, str]]:
    samples = []
    for source in sources:
        results = source / "results.csv"
        if results.exists():
            samples += [(source.name, source / row.file, row.expected) for row in pd.read_csv(results).itertuples()]
            continue
        for folder in sorted(p for p in source.iterdir() if p.is_dir()):
            emotion = FOLDER_NAMES.get(folder.name.lower())
            if emotion is None:
                print(f"Пропущено папку «{folder.name}»: назва не збігається з жодною емоцією")
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
            samples += [(source.name, f, emotion) for f in files]
    return samples


def evaluate(samples: list[tuple[str, Path, str]], model: str, denoise: bool) -> pd.DataFrame:
    classifier = VoiceEmotionClassifier(model)
    classifier.warm_up()
    detector = SpeechDetector(denoise=denoise)
    rows = []
    for source, path, expected in samples:
        segment = detector.process(load_audio(path))
        result = classifier.predict(segment.speech_span(), 0)
        rows.append({
            "source": source,
            "file": path.name,
            "expected": expected,
            "predicted": EMOTIONS[int(result.probs.argmax())] if result.probs is not None else "no_speech",
            "speech_s": round(len(segment.speech_span()) / SAMPLE_RATE, 2),
            "processing_ms": round(result.processing_ms, 1),
            **({e: round(float(p), 4) for e, p in zip(EMOTIONS, result.probs)} if result.probs is not None else {}),
        })
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Точність емоцій за голосом: записи tools/voice_demo.py --scenario "
                    "або папка з підпапками за емоціями (злість, радість, ...)")
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--models", nargs="+",
                        default=["emotion2vec/emotion2vec_plus_large", "emotion2vec/emotion2vec_plus_base"])
    parser.add_argument("--no-denoise", action="store_true")
    args = parser.parse_args()

    samples = collect_samples(args.sources)
    print(f"Записів: {len(samples)}; за емоціями: {pd.Series([s[2] for s in samples]).value_counts().to_dict()}")
    out_dir = args.sources[0]
    summary = []
    for model in args.models:
        result = evaluate(samples, model, not args.no_denoise)
        name = model.split("/")[-1]
        suffix = "_nodenoise" if args.no_denoise else ""
        result.to_csv(out_dir / f"eval_{name}{suffix}.csv", index=False)
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
    summary_path = out_dir / f"eval_summary{suffix}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nЗбережено: {summary_path}")


if __name__ == "__main__":
    main()
