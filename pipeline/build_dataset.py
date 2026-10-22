"""Merge simulator tasks and council output into train/val files for fine-tuning.

  python build_dataset.py --sim 4000
  python build_dataset.py --sim 4000 --extra ../data/paraphrased.jsonl ../data/invented.jsonl
  # after testing your trained model, focus new data on what it got wrong:
  python build_dataset.py --sim 4000 --focus ../data/leaderboard.json --focus-model student
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from council import DATA, EVAL_SEED
from records import read_jsonl, scenario_to_record, to_sft_examples
from scenarios import FAMILIES, generate


def focus_weights(board_path: Path, model: str) -> dict[str, float]:
    """More examples of the task types the model is weak at."""
    board = json.loads(board_path.read_text())["models"][model]["by_family"]
    return {f: 1.0 + 3.0 * (1.0 - acc) for f, acc in board.items() if f in FAMILIES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", type=int, default=4000, help="simulated tasks to generate")
    ap.add_argument("--extra", nargs="*", type=Path, default=[], help="council JSONL files to include")
    ap.add_argument("--focus", type=Path, help="leaderboard.json from testing your trained model")
    ap.add_argument("--focus-model", default="student")
    ap.add_argument("--ar-ratio", type=float, default=0.5, help="share of Arabic requests")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DATA)
    a = ap.parse_args()
    if a.seed == EVAL_SEED:
        raise SystemExit(f"seed {EVAL_SEED} is reserved for the leaderboard")

    weights = focus_weights(a.focus, a.focus_model) if a.focus else None
    if weights:
        print("Focus weights:", {k: round(v, 1) for k, v in sorted(weights.items())})
    records = [scenario_to_record(s) for s in generate(a.sim, seed=a.seed, weights=weights, ar_ratio=a.ar_ratio)]
    for path in a.extra:
        if path.exists():
            extra = list(read_jsonl(path))
            print(f"{path.name}: {len(extra)} tasks")
            records += extra
        else:
            print(f"skipping {path} (not found)")

    rng = random.Random(a.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * a.val))
    a.out.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    for name, part in (("val", records[:n_val]), ("train", records[n_val:])):
        with open(a.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in part:
                for ex in to_sft_examples(rec):
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    counts[name] += 1
    by_source = collections.Counter(r["source"].split(":")[0] for r in records)
    print(f"{len(records)} tasks ({dict(by_source)}) -> {counts['train']} train / {counts['val']} val steps in {a.out}")


if __name__ == "__main__":
    main()
