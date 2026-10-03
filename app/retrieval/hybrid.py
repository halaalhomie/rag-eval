"""Hybrid retrieval: dense + BM25, fused.

Each retriever returns RETRIEVAL_CANDIDATES results; the two lists are fused and the top-k
returned. Two fusion methods (FUSION_METHOD):

- rrf (default): weighted Reciprocal Rank Fusion (Cormack et al., 2009)
      score(d) = Σ_r w_r / (RRF_K + rank_r(d))
  Uses ranks only, so the two retrievers' incompatible score scales (cosine similarity vs
  unbounded BM25) never need reconciling. A document absent from a list contributes 0.
- linear: Σ_r w_r · minmax_r(score_r(d)), each list min-max normalized over its own
  candidates. It is sensitive to score distributions, and included as the comparison.

Every fused result records its rank and score in each component list (`components`), so
it is observable why a chunk ranked where it did.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import FusionMethod
from app.retrieval.base import MetadataFilter, RetrievalResult, Retriever


def rrf_fuse(
    ranked_lists: dict[str, Sequence[RetrievalResult]],
    weights: dict[str, float],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for name, results in ranked_lists.items():
        w = weights.get(name, 0.0)
        for r in results:
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + w / (k + r.rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def linear_fuse(
    ranked_lists: dict[str, Sequence[RetrievalResult]], weights: dict[str, float]
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for name, results in ranked_lists.items():
        if not results:
            continue
        values = [r.score for r in results]
        lo, hi = min(values), max(values)
        w = weights.get(name, 0.0)
        for r in results:
            norm = (r.score - lo) / (hi - lo) if hi > lo else 1.0
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + w * norm
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        retrievers: dict[str, Retriever],
        weights: dict[str, float],
        *,
        method: FusionMethod = FusionMethod.RRF,
        rrf_k: int = 60,
        candidates: int = 50,
    ):
        if not retrievers or sum(weights.get(n, 0.0) for n in retrievers) <= 0:
            raise ValueError("hybrid retrieval needs at least one retriever with weight > 0")
        self.retrievers = retrievers
        self.weights = weights
        self.method = method
        self.rrf_k = rrf_k
        self.candidates = candidates

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievalResult]:
        if k <= 0 or not query.strip():
            return []
        depth = max(k, self.candidates)
        lists = {name: r.retrieve(query, depth, filters) for name, r in self.retrievers.items()}
        if self.method is FusionMethod.RRF:
            fused = rrf_fuse(lists, self.weights, self.rrf_k)
        else:
            fused = linear_fuse(lists, self.weights)

        by_id: dict[str, RetrievalResult] = {}
        components: dict[str, dict[str, dict[str, float]]] = {}
        for name, results in lists.items():
            for r in results:
                by_id.setdefault(r.chunk_id, r)
                components.setdefault(r.chunk_id, {})[name] = {
                    "rank": r.rank,
                    "score": round(r.score, 6),
                }
        return [
            by_id[chunk_id].model_copy(
                update={
                    "score": score,
                    "rank": rank,
                    "retriever": self.name,
                    "components": components[chunk_id],
                }
            )
            for rank, (chunk_id, score) in enumerate(fused[:k], start=1)
        ]
