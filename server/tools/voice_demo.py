import argparse
import csv
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.audio.asr import SpeechRecognizer, UtteranceBuffer
from app.audio.features import voice_features
from app.audio.preprocessing import SAMPLE_RATE, SpeechDetector
from app.audio.voice_emotion import VoiceEmotionClassifier
from app.config import load_settings
from app.video.face_emotion import EMOTIONS, EMOTIONS_UK

PHRASES = [
    "Давайте ще раз обговоримо умови договору.",
    "Ви пропонуєте саме таку ціну?",
]
SCENARIO = [
    ("neutral", "рівно і спокійно, як звичайне ділове речення"),
    ("happy", "з посмішкою, задоволено, ніби почули гарну новину"),
    ("angry", "роздратовано, з натиском, трохи голосніше і різкіше"),
    ("sad", "розчаровано, тихіше і повільніше, ніби угода зривається"),
    ("fear", "тривожно і невпевнено, з запинками, ніби боїтеся відмови"),
    ("surprise", "здивовано, з підйомом тону, ніби не очікували такого"),
]


def describe(features: dict, probs: np.ndarray | None) -> str:
    emotion = f"{EMOTIONS_UK[EMOTIONS[int(probs.argmax())]]} {probs.max():.2f}" if probs is not None else "—"
    if "f0_mean_hz" not in features:
        return f"мови немає (частка мови {features['speech_ratio']:.2f})"
    return (f"емоція: {emotion:16s} тон {features['f0_mean_hz']:5.0f} Гц ±{features['f0_std_st']:.1f} пт, "
            f"гучність {features['loudness_db']:5.1f} дБ, темп {features['speech_rate']:.1f}/с, "
            f"паузи {features['pause_ratio']:.2f}")


def run_live(detector, classifier, asr, settings, chunk_s: float) -> None:
    chunks: queue.Queue = queue.Queue()
    utterances = UtteranceBuffer(settings.asr.min_pause_s, settings.asr.max_utterance_s)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=int(SAMPLE_RATE * chunk_s),
                        callback=lambda data, *_: chunks.put(data[:, 0].copy())):
        print("Говоріть. Вихід: Ctrl+C.")
        while True:
            segment = detector.process(chunks.get())
            features = voice_features(segment)
            result = classifier.predict(segment.speech_span(), 0)
            print(describe(features, result.probs))
            utterance = utterances.add(segment.speech_span(), segment.trailing_silence_s(), 0)
            if utterance is not None and asr is not None:
                transcript = asr.transcribe(utterance[0], 0)
                print(f"    текст ({transcript.processing_ms:.0f} мс): {transcript.text or '—'}")


def run_scenario(detector, classifier, asr, record_s: float, rounds: int) -> None:
    out_dir = SERVER_DIR / "recordings" / f"voice_demo_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    rows = []
    total = rounds * len(SCENARIO)
    print(f"Запис у {out_dir}\nКожну фразу вимовляйте після слова «ГОВОРІТЬ!», на запис {record_s:.0f} с.")
    for round_index in range(rounds):
        phrase = PHRASES[round_index % len(PHRASES)]
        for expected, manner in SCENARIO:
            print("\n" + "=" * 70)
            print(f"  {len(rows) + 1}/{total}   ЕМОЦІЯ:  {EMOTIONS_UK[expected].upper()}")
            print(f"  Як говорити: {manner}")
            print(f"  Фраза: «{phrase}»")
            print("=" * 70)
            input("Натисніть Enter, коли будете готові...")
            print(">>> ГОВОРІТЬ!  <<<")
            audio = sd.rec(int(record_s * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            print("Стоп, запис завершено.")
            audio = audio[:, 0]
            name = f"{len(rows):02d}_{expected}.wav"
            sf.write(out_dir / name, audio, SAMPLE_RATE, subtype="PCM_16")
            segment = detector.process(audio)
            features = voice_features(segment)
            result = classifier.predict(segment.speech_span(), 0)
            transcript = asr.transcribe(segment.speech_span(), 0).text if asr and segment.has_speech else ""
            print(describe(features, result.probs))
            print(f"    текст: {transcript or '—'}")
            rows.append({
                "file": name, "expected": expected, "phrase": phrase, "transcript": transcript,
                "predicted": EMOTIONS[int(result.probs.argmax())] if result.probs is not None else "no_speech",
                **({e: round(float(p), 4) for e, p in zip(EMOTIONS, result.probs)} if result.probs is not None else {}),
                **{k: v for k, v in features.items() if k != "mfcc"},
            })
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    correct = sum(r["expected"] == r["predicted"] for r in rows)
    print(f"\nВгадано {correct} з {len(rows)}. Результати: {out_dir / 'results.csv'}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Аналіз голосу з мікрофона ноутбука")
    parser.add_argument("--scenario", action="store_true", help="записати фрази з різними емоціями")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--record-s", type=float, default=5.0)
    parser.add_argument("--chunk-s", type=float, default=2.0)
    parser.add_argument("--no-asr", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    audio = settings.audio
    started = time.perf_counter()
    print("Завантаження моделі емоцій голосу (emotion2vec+, ~30–50 с)...", flush=True)
    detector = SpeechDetector(audio.vad_aggressiveness, audio.vad_frame_ms, audio.min_rms_db, denoise=audio.denoise)
    classifier = VoiceEmotionClassifier(audio.voice_model, audio.voice_device, audio.min_speech_s)
    classifier.warm_up()
    asr = None
    if not args.no_asr:
        print("Завантаження розпізнавання мовлення (whisper)...", flush=True)
        s = settings.asr
        asr = SpeechRecognizer(s.model, s.language, s.device, s.compute_type, s.beam_size)
        asr.warm_up()
    print(f"Готово за {time.perf_counter() - started:.0f} с.")
    try:
        if args.scenario:
            run_scenario(detector, classifier, asr, args.record_s, args.rounds)
        else:
            run_live(detector, classifier, asr, settings, args.chunk_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
