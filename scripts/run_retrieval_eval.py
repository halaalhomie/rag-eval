"""Evaluate a retriever on the evaluation dataset.

Usage:
    python scripts/run_retrieval_eval.py --retriever dense --split test
    python scripts/run_retrieval_eval.py --retriever dense --split dev --k 1,5,10
    BM25_STEMMING=false python scripts/run_retrieval_eval.py --retriever bm25 --split dev \
        --label no-stem                        # ablations via settings env vars

Writes data/experiments/<run-id>/report.json (committed) and per_query.jsonl (gitignored).
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.db.session import session_scope
from app.evaluation.datasets.labels import ChunkIndex
from app.evaluation.datasets.schema import dataset_version, load_dataset
from app.evaluation.retrieval.runner import evaluate_retrieval
from app.rag.strategies import Components, build_hybrid, build_hybrid_rerank
from app.retrieval.bm25 import build_bm25
from app.retrieval.dense import DenseRetriever
from app.retrieval.embeddings import get_embedder
from app.utils.logging import configure_logging


def build_retriever(name: str, settings):
    if name == "dense":
        return DenseRetriever(
            get_embedder(settings.embedding), settings.retrieval.similarity_metric
        )
    if name == "bm25":
        return build_bm25(settings)
    # Retrieval-only evaluation: the LLM slot is never called.
    components = Components(llm=None, embedder=get_embedder(settings.embedding))
    if name == "hybrid":
        return build_hybrid(settings, components)
    if name == "hybrid_rerank":
        return build_hybrid_rerank(settings, components)
    raise SystemExit(f"unknown retriever '{name}' (available: dense, bm25, hybrid, hybrid_rerank)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/kubernetes_v1.jsonl"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--retriever", default="dense")
    parser.add_argument("--k", default="1,3,5,10,20")
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--label", help="suffix for the run id, e.g. an ablation name")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.observability.log_level)
    split = None if args.split == "all" else args.split
    items = load_dataset(args.dataset, split=split)
    with session_scope() as s:
        index = ChunkIndex.load(s)

    report, per_query = evaluate_retrieval(
        build_retriever(args.retriever, settings),
        items,
        index,
        dataset=args.dataset.stem,
        dataset_version=dataset_version(args.dataset),
        split=args.split,
        ks=[int(k) for k in args.k.split(",")],
        config=settings.snapshot(),
    )
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-retrieval-{args.retriever}-{args.split}"
    if args.label:
        run_id += f"-{args.label}"
    out = args.out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    with (out / "per_query.jsonl").open("w") as f:
        for r in per_query:
            f.write(r.model_dump_json() + "\n")
    headline = {
        k: report.overall[k] for k in report.overall if k.endswith(("@5", "@10")) or k == "mrr"
    }
    print(json.dumps({"run": run_id, "n": report.n_evaluated, **headline}, indent=2))
