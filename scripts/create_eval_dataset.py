"""Generate the synthetic evaluation dataset from the ingested corpus.

Usage:
    python scripts/create_eval_dataset.py                      # full run (~330 items)
    python scripts/create_eval_dataset.py --pilot 3 --out /tmp/pilot.jsonl
    python scripts/create_eval_dataset.py --extend --seed 43 --attempt-factor 12 \\
        --quota semantic=20 --quota multi_hop=16             # top up an existing dataset

Requires the LLM server (see README) and an embedded corpus. Writes:
    data/eval/<name>.jsonl       one EvalItem per line
    data/eval/<name>.meta.json   generator, corpus commit, seeds, counts, rejection stats
"""

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Document
from app.db.session import session_scope
from app.evaluation.datasets.generator import (
    DatasetGenerator,
    assign_splits,
    nonexistent_ground_truth,
)
from app.evaluation.datasets.schema import DatasetMeta, load_dataset, save_dataset
from app.generation.factory import build_llm
from app.ingestion.chunking import StructureAwareChunker
from app.retrieval.dense import DenseRetriever
from app.retrieval.embeddings import get_embedder
from app.utils.logging import configure_logging

QUOTAS = {
    "simple_factual": 60,
    "semantic": 45,
    "keyword_heavy": 45,
    "numerical": 30,
    "multi_hop": 40,
    "comparison": 30,
    "ambiguous": 25,
    "adversarial": 25,
    "unanswerable/nonexistent_feature": 15,
    "unanswerable/out_of_domain": 15,
}


def merge_stats(old: dict, new: dict) -> dict:
    merged = {k: dict(v) for k, v in old.items()}
    for key, stats in new.items():
        target = merged.setdefault(key, {})
        for name, n in stats.items():
            target[name] = target.get(name, 0) + n
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="kubernetes_v1")
    parser.add_argument("--out", type=Path, help="override output .jsonl path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--pilot", type=int, help="generate N items per type only")
    parser.add_argument("--extend", action="store_true", help="add items to an existing set")
    parser.add_argument("--quota", action="append", default=[], help="type=N (repeatable)")
    parser.add_argument("--attempt-factor", type=int, default=6)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.observability.log_level)
    out = args.out or Path("data/eval") / f"{args.name}.jsonl"
    meta_path = out.with_suffix(".meta.json")

    if args.quota:
        quotas = {k: int(v) for k, v in (q.split("=", 1) for q in args.quota)}
        unknown = set(quotas) - set(QUOTAS)
        if unknown:
            parser.error(f"unknown types {sorted(unknown)}; valid: {sorted(QUOTAS)}")
    elif args.pilot:
        quotas = {k: args.pilot for k in QUOTAS}
    else:
        quotas = QUOTAS

    existing = load_dataset(out) if args.extend else []
    old_meta = json.loads(meta_path.read_text()) if args.extend else {}

    retriever = DenseRetriever(
        get_embedder(settings.embedding), settings.retrieval.similarity_metric
    )
    gen = DatasetGenerator(
        build_llm(settings.llm), settings.llm.llm_fast_model, session_scope, retriever, args.seed
    )
    gen.seed_existing(existing)
    started = datetime.now(UTC)
    new_items = gen.generate(quotas, max_attempt_factor=args.attempt_factor)

    items = existing + new_items
    # Ground-truth wording for nonexistent features changed after the first run (a fake
    # flag can sit next to a real one); re-derive it so every item uses the same rule.
    for item in items:
        if item.subtype == "nonexistent_feature" and "fake_term" in item.notes:
            item.ground_truth_answer = nonexistent_ground_truth(item.notes["fake_term"])
    assign_splits(items, args.dev_fraction, old_meta.get("seed", args.seed))
    items.sort(key=lambda i: (i.question_type.value, i.id))
    version = save_dataset(items, out)

    with session_scope() as s:
        commit = s.scalar(select(Document.metadata_["corpus_commit"].astext).limit(1))
    stats = {k: v.as_dict() for k, v in gen.stats.items()}
    meta = DatasetMeta(
        name=args.name,
        version=version,
        created_at=old_meta.get("created_at", started.isoformat()),
        corpus="kubernetes",
        corpus_commit=commit,
        chunker_fingerprint=StructureAwareChunker(settings.chunking).fingerprint(),
        generator_model=settings.llm.llm_fast_model,
        seed=old_meta.get("seed", args.seed),
        counts=dict(Counter(i.question_type.value for i in items)),
        split_counts=dict(Counter(i.split for i in items)),
        generation_stats=merge_stats(old_meta.get("generation_stats", {}), stats),
        runs=[
            *old_meta.get("runs", []),
            {"started_at": started.isoformat(), "seed": args.seed, "added": len(new_items)},
        ],
        description=(
            "SYNTHETIC: questions, answers and evidence selections generated by a local LLM "
            "and filtered by deterministic validators. Not human-annotated."
        ),
    )
    meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Wrote %d items (%d new, version %s) to %s", len(items), len(new_items), version, out
    )
    print(
        json.dumps(
            {
                "items": len(items),
                "new": len(new_items),
                **meta.split_counts,
                "counts": meta.counts,
            },
            indent=2,
        )
    )
