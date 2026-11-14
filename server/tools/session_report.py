import argparse
import collections
import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Короткий звіт про сесію за журналом JSONL")
    parser.add_argument("log", type=Path, nargs="?", help="типово — найновіший журнал у logs/")
    args = parser.parse_args()
    path = args.log or max((SERVER_DIR / "logs").glob("2*.jsonl"), key=lambda p: p.stat().st_mtime)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    start = rows[0]
    print(f"{path.name}: режим {start['mode']}, зразок голосу {start.get('voice_profile')}")
    print(f"контекст: {start.get('context') or '—'}")
    print(dict(collections.Counter(r["event"] for r in rows)))
    for r in rows:
        if r["event"] == "transcript":
            print(f"  [{r.get('speaker', '?'):7s} {r.get('similarity')}] «{r['text']}» "
                  f"({r['duration_s']} с, {r['processing_ms']:.0f} мс)")
        elif r["event"] == "hint":
            print(f"  >>> {r.get('trigger', 'state')}/{r['state']}/{r['source']} {r['latency_ms']} мс"
                  f"{' на «' + r['heard'] + '»' if r.get('heard') else ''}\n      {r['reply']} ({r['tone']})")
        elif r["event"] == "session_stop":
            print(f"  стоп: кадрів {r['frames']}, проаналізовано {r['frames_analyzed']}, аудіо {r['audio_seconds']} с")
    voice = [r for r in rows if r["event"] == "voice" and r.get("speaker")]
    if voice:
        print("мовці у фрагментах:", dict(collections.Counter(r["speaker"] for r in voice)))
        sims = sorted(r["similarity"] for r in voice if r.get("similarity") is not None)
        if sims:
            print("схожість зі зразком:", [round(s, 2) for s in sims])
        print("підсилення, дБ:", [r["features"].get("gain_db") for r in voice][:12])


if __name__ == "__main__":
    main()
