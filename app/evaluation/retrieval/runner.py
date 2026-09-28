"""Retrieval evaluation: run a retriever over eval items and score its rankings.

Per item: retrieve max(K) results once, derive the relevant chunks from the item's evidence
spans against the current index, and compute every metric at every K from that one
ranking. Items with no evidence (unanswerable) are excluded from retrieval metrics and
reported as excluded; they are scored by the generation evaluation instead.
"""

from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.datasets.labels import ChunkIndex
from app.evaluation.datasets.schema import EvalItem
from app.evaluation.metrics import retrieval as M
from app.retrieval.base import Retriever

DEFAULT_KS = (1, 3, 5, 10, 20)
CI_METRICS = ("evidence_recall@5", "evidence_recall@10", "mrr", "ndcg@10")
BOOTSTRAP_SAMPLES = 2000


class QueryResult(BaseModel):
    item_id: str
    question_type: str
    difficulty: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    evidence_spans: int
    unmapped_spans: int  # evidence spans that no current chunk covers
    latency_s: float
    metrics: dict[str, float]


class RetrievalReport(BaseModel):
    retriever: str
    dataset: str
    dataset_version: str
    split: str | None
    created_at: str
    ks: list[int]
    n_evaluated: int
    n_excluded_no_evidence: int
    unmapped_evidence_spans: int
    overall: dict[str, float]
    # Percentile bootstrap over queries (seeded): [low, high] of the 95% interval.
    overall_ci95: dict[str, list[float]] = Field(default_factory=dict)
    by_type: dict[str, dict[str, float]]
    counts_by_type: dict[str, int]
    latency_s: dict[str, float]
    config: dict[str, Any] = Field(default_factory=dict)


def score_ranking(
    retrieved: Sequence[str],
    retrieved_docs: Sequence[str],
    relevant: set[str],
    evidence_chunks: list[set[str]],
    relevant_docs: set[str],
    ks: Sequence[int],
) -> dict[str, float]:
    m: dict[str, float] = {"mrr": M.reciprocal_rank(retrieved, relevant, max(ks))}
    # Document ranking = first occurrence order of each document in the chunk ranking.
    doc_ranking = list(dict.fromkeys(retrieved_docs))
    for k in ks:
        m[f"evidence_recall@{k}"] = M.evidence_recall_at_k(retrieved, evidence_chunks, k)
        m[f"hit@{k}"] = M.hit_at_k(retrieved, relevant, k) if relevant else 0.0
        m[f"precision@{k}"] = M.precision_at_k(retrieved, relevant, k)
        m[f"recall@{k}"] = M.recall_at_k(retrieved, relevant, k) if relevant else 0.0
        m[f"ndcg@{k}"] = M.ndcg_at_k(retrieved, relevant, k) if relevant else 0.0
        m[f"doc_recall@{k}"] = len(relevant_docs & set(doc_ranking[:k])) / len(relevant_docs)
    return m


def bootstrap_ci(
    values: Sequence[float], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> list[float]:
    """95% percentile-bootstrap interval of the mean. With ~200 queries (and 13-42 per
    question type) point estimates alone would overstate how precise these numbers are."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(samples))
    return [round(means[int(0.025 * samples)], 4), round(means[int(0.975 * samples) - 1], 4)]


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {k: round(statistics.fmean(r[k] for r in rows), 4) for k in keys}


def evaluate_retrieval(
    retriever: Retriever,
    items: list[EvalItem],
    index: ChunkIndex,
    *,
    dataset: str,
    dataset_version: str,
    split: str | None,
    ks: Sequence[int] = DEFAULT_KS,
    config: dict[str, Any] | None = None,
) -> tuple[RetrievalReport, list[QueryResult]]:
    ks = sorted(set(ks))
    depth = max(ks)
    scored = [i for i in items if i.has_evidence]
    retriever.retrieve("warm up", 1)  # exclude model loading from latency

    results: list[QueryResult] = []
    for item in scored:
        started = time.perf_counter()
        hits = retriever.retrieve(item.question, depth)
        latency = time.perf_counter() - started
        evidence_chunks = index.evidence_chunks(item)
        relevant = set().union(*evidence_chunks)
        retrieved = [h.chunk_id for h in hits]
        results.append(
            QueryResult(
                item_id=item.id,
                question_type=item.question_type.value,
                difficulty=item.difficulty,
                retrieved_chunk_ids=retrieved,
                relevant_chunk_ids=sorted(relevant),
                evidence_spans=len(evidence_chunks),
                unmapped_spans=sum(not c for c in evidence_chunks),
                latency_s=round(latency, 5),
                metrics=score_ranking(
                    retrieved,
                    [h.document_id for h in hits],
                    relevant,
                    evidence_chunks,
                    set(item.relevant_document_ids),
                    ks,
                ),
            )
        )

    by_type_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in results:
        by_type_rows[r.question_type].append(r.metrics)
    latencies = sorted(r.latency_s for r in results)
    report = RetrievalReport(
        retriever=retriever.name,
        dataset=dataset,
        dataset_version=dataset_version,
        split=split,
        created_at=datetime.now(UTC).isoformat(),
        ks=list(ks),
        n_evaluated=len(results),
        n_excluded_no_evidence=len(items) - len(scored),
        unmapped_evidence_spans=sum(r.unmapped_spans for r in results),
        overall=_mean([r.metrics for r in results]),
        overall_ci95={
            m: bootstrap_ci([r.metrics[m] for r in results])
            for m in CI_METRICS
            if results and m in results[0].metrics
        },
        by_type={t: _mean(rows) for t, rows in sorted(by_type_rows.items())},
        counts_by_type={t: len(rows) for t, rows in sorted(by_type_rows.items())},
        latency_s={
            "mean": round(statistics.fmean(latencies), 5) if latencies else 0.0,
            "p50": round(latencies[len(latencies) // 2], 5) if latencies else 0.0,
            "p95": round(latencies[int(len(latencies) * 0.95) - 1], 5) if latencies else 0.0,
        },
        config=config or {},
    )
    return report, results
