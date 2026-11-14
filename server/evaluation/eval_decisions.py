import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.config import load_settings
from app.decision import build_decisions
from app.decision.base import STRATEGIES_UK, Decision, StateVector


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Порівняння модулів рішень А, Б і гібрид на типових ситуаціях")
    parser.add_argument("--scenarios", type=Path, default=SERVER_DIR / "evaluation" / "scenarios_uk.json")
    parser.add_argument("--modes", nargs="+", default=["fuzzy", "llm", "hybrid"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    settings = load_settings()
    print("Завантаження модулів (прогрів мовної моделі)...", flush=True)
    modules = build_decisions(settings)
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    out_dir = SERVER_DIR / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"decisions_{datetime.now():%Y%m%d_%H%M%S}.csv"

    rows = []
    for repeat in range(args.repeats):
        for scenario in scenarios:
            state = StateVector(scenario["state"], scenario["degrees"], scenario["confidence"])
            for mode in args.modes:
                history: list[Decision] = []
                started = time.perf_counter()
                decision = modules[mode].decide(state, scenario["voice"], scenario["transcript"], history)
                elapsed = (time.perf_counter() - started) * 1000
                rows.append({
                    "repeat": repeat, "scenario": scenario["id"], "state": scenario["state"],
                    "transcript": scenario["transcript"], "mode": mode, "source": decision.source,
                    "strategy": STRATEGIES_UK.get(decision.strategy, "") if decision.strategy else "",
                    "reply": decision.reply, "tone": decision.tone, "ms": round(elapsed),
                    "error": decision.details.get("error", ""),
                })
                print(f"[{scenario['id']:18s} {mode:6s} {elapsed:6.0f} мс {decision.source:15s}] "
                      f"{decision.reply} (тон: {decision.tone})", flush=True)

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    summary = df.groupby("mode").agg(
        calls=("ms", "size"),
        median_ms=("ms", "median"),
        p90_ms=("ms", lambda s: s.quantile(0.9)),
        max_ms=("ms", "max"),
        fallback_share=("source", lambda s: s.str.endswith("fallback").mean()),
    ).round(2)
    print("\n", summary.to_string(), sep="")
    summary.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), encoding="utf-8-sig")
    print(f"\nЗбережено: {out_path}")


if __name__ == "__main__":
    main()
