"""Export a stratified random sample of eval items to CSV for human review.

The dataset is synthetic. This sheet is how a person checks it: fill in `verdict`
(correct / incorrect / unclear) and optionally `comment`, then report the agreement rate
per question type. Only verdicts entered by a person should be described as human review.

Usage:
    python scripts/export_review_sheet.py --per-type 5 --out data/eval/review_sample.csv
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from app.evaluation.datasets.schema import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/kubernetes_v1.jsonl"))
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/eval/review_sample.csv"))
    args = parser.parse_args()

    by_type = defaultdict(list)
    for item in load_dataset(args.dataset):
        by_type[item.question_type.value].append(item)
    rng = random.Random(args.seed)
    rows = []
    for qtype, items in sorted(by_type.items()):
        for item in rng.sample(items, min(args.per_type, len(items))):
            rows.append(
                {
                    "id": item.id,
                    "question_type": qtype,
                    "question": item.question,
                    "ground_truth_answer": item.ground_truth_answer,
                    "evidence": " || ".join(e.quote for e in item.evidence),
                    "notes": "; ".join(f"{k}={v}" for k, v in item.notes.items()),
                    "verdict": "",
                    "comment": "",
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} items to {args.out}")
