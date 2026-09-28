"""Paired comparison of two retrieval runs on the same questions.

Both runs must cover the same items (same dataset version and split). The difference in a
metric is bootstrapped per query (paired), which is far tighter and more honest than
comparing two independent confidence intervals.

Usage:
    python scripts/compare_retrieval_runs.py data/experiments/<run-a> data/experiments/<run-b> \
        --metric evidence_recall@5
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(run: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads((run / "report.json").read_text())
    rows = {}
    for line in (run / "per_query.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["item_id"]] = r
    return report, rows


def paired_bootstrap(diffs: list[float], samples: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(samples))
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--metric", action="append", help="repeatable; default: headline set")
    args = parser.parse_args()
    metrics = args.metric or ["evidence_recall@5", "evidence_recall@10", "mrr", "ndcg@10"]

    rep_a, a = load(args.run_a)
    rep_b, b = load(args.run_b)
    if rep_a["dataset_version"] != rep_b["dataset_version"] or set(a) != set(b):
        raise SystemExit("runs are not comparable: different dataset version or items")

    out = {"a": args.run_a.name, "b": args.run_b.name, "n": len(a), "metrics": {}}
    for m in metrics:
        diffs = [b[i]["metrics"][m] - a[i]["metrics"][m] for i in sorted(a)]
        by_type = defaultdict(list)
        for i in a:
            by_type[a[i]["question_type"]].append(b[i]["metrics"][m] - a[i]["metrics"][m])
        out["metrics"][m] = {
            "a": round(statistics.fmean(a[i]["metrics"][m] for i in a), 4),
            "b": round(statistics.fmean(b[i]["metrics"][m] for i in b), 4),
            "diff_b_minus_a": round(statistics.fmean(diffs), 4),
            "diff_ci95": [round(x, 4) for x in paired_bootstrap(diffs)],
            "b_better": sum(d > 0 for d in diffs),
            "a_better": sum(d < 0 for d in diffs),
            "tied": sum(d == 0 for d in diffs),
            "diff_by_type": {t: round(statistics.fmean(v), 3) for t, v in sorted(by_type.items())},
        }
    print(json.dumps(out, indent=2))
