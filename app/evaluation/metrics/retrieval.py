"""Retrieval metrics (deterministic, binary relevance).

Standard IR definitions over a ranked list of retrieved IDs and a set of relevant IDs:

- Recall@K    = |relevant ∩ top-K| / |relevant|
- Precision@K = |relevant ∩ top-K| / K
- Hit@K       = 1 if any relevant item is in the top-K
- MRR         = 1 / rank of the first relevant item (0 if none); averaged over queries
- NDCG@K      = DCG@K / IDCG@K with DCG = Σ rel_i / log2(i + 1) and binary rel_i;
                IDCG places min(|relevant|, K) relevant items at the top

Chunk-level recall has a subtlety with overlapping chunks: one evidence span can be covered
by two chunks, and a system that retrieves just one of them has all the evidence yet only
50% chunk recall. So we also report:

- Evidence Recall@K = fraction of evidence spans with at least one covering chunk in the
  top-K. This is the primary recall metric, especially for multi-hop items that need two
  spans.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(relevant & set(retrieved[:k])) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    return len(relevant & set(retrieved[:k])) / k


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    return float(any(r in relevant for r in retrieved[:k]))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    for i, r in enumerate(retrieved[:k] if k else retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _check_k(k)
    if not relevant:
        raise ValueError("NDCG is undefined without relevant items")
    seen: set[str] = set()
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant and r not in seen:  # duplicates earn no extra gain
            dcg += 1.0 / math.log2(i + 1)
            seen.add(r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evidence_recall_at_k(
    retrieved: Sequence[str], evidence_chunks: Sequence[set[str]], k: int
) -> float:
    """Fraction of evidence spans covered by at least one top-K chunk.

    A span with no covering chunk in the index (e.g. lost by a chunking change) counts as
    not found; that is a real failure of the index, not something to skip.
    """
    _check_k(k)
    if not evidence_chunks:
        raise ValueError("evidence recall is undefined without evidence")
    top = set(retrieved[:k])
    return sum(bool(chunks & top) for chunks in evidence_chunks) / len(evidence_chunks)
