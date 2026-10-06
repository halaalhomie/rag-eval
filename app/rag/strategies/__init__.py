"""Strategy registry: maps RAG_STRATEGY values to pipeline builders.

A strategy is registered only once it is implemented, so a request for anything else fails
loudly instead of silently falling back to a different pipeline (which would corrupt
experiment comparisons).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.config import RagStrategy, Settings
from app.db.session import session_scope
from app.generation.llm import LLMClient
from app.rag.strategies.baseline import BaselineRAG
from app.rag.types import RAGResult
from app.retrieval.base import MetadataFilter
from app.retrieval.bm25 import build_bm25
from app.retrieval.dense import DenseRetriever, SessionFactory
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankingRetriever, get_reranker


class Pipeline(Protocol):
    name: str

    def run(
        self, query: str, *, top_k: int | None = None, filters: MetadataFilter | None = None
    ) -> RAGResult: ...


class StrategyNotImplementedError(ValueError):
    pass


@dataclass(frozen=True)
class Components:
    """Heavy, shareable dependencies (model weights, HTTP clients)."""

    llm: LLMClient
    embedder: Embedder
    session_factory: SessionFactory = field(default=session_scope)


def _baseline(settings: Settings, c: Components) -> Pipeline:
    return BaselineRAG(
        DenseRetriever(c.embedder, settings.retrieval.similarity_metric, c.session_factory),
        c.llm,
        settings,
    )


def _bm25(settings: Settings, c: Components) -> Pipeline:
    return BaselineRAG(build_bm25(settings, c.session_factory), c.llm, settings, name="bm25")


def build_hybrid(settings: Settings, c: Components) -> HybridRetriever:
    r = settings.retrieval
    return HybridRetriever(
        {
            "dense": DenseRetriever(c.embedder, r.similarity_metric, c.session_factory),
            "bm25": build_bm25(settings, c.session_factory),
        },
        {"dense": r.dense_weight, "bm25": r.bm25_weight},
        method=r.fusion_method,
        rrf_k=r.rrf_k,
        candidates=r.retrieval_candidates,
    )


def _hybrid(settings: Settings, c: Components) -> Pipeline:
    return BaselineRAG(build_hybrid(settings, c), c.llm, settings, name="hybrid")


def build_hybrid_rerank(settings: Settings, c: Components) -> RerankingRetriever:
    return RerankingRetriever(
        build_hybrid(settings, c),
        get_reranker(settings),
        candidates=settings.retrieval.rerank_candidates,
        name="hybrid_rerank",
    )


def _hybrid_rerank(settings: Settings, c: Components) -> Pipeline:
    return BaselineRAG(
        build_hybrid_rerank(settings, c),
        c.llm,
        settings,
        name="hybrid_rerank",
        default_top_k=settings.retrieval.rerank_top_k,
    )


BUILDERS = {
    RagStrategy.BASELINE: _baseline,
    # "dense" is experiment A (dense RAG); today it is the baseline pipeline. It gets its
    # own builder only if the two diverge.
    RagStrategy.DENSE: _baseline,
    RagStrategy.BM25: _bm25,
    RagStrategy.HYBRID: _hybrid,
    RagStrategy.HYBRID_RERANK: _hybrid_rerank,
}


def implemented() -> list[str]:
    return [s.value for s in BUILDERS]


def build_pipeline(strategy: RagStrategy, settings: Settings, components: Components) -> Pipeline:
    builder = BUILDERS.get(strategy)
    if builder is None:
        raise StrategyNotImplementedError(
            f"strategy '{strategy.value}' is not implemented yet (available: {implemented()})"
        )
    return builder(settings, components)
