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
from app.retrieval.dense import DenseRetriever, SessionFactory
from app.retrieval.embeddings import Embedder


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


BUILDERS = {
    RagStrategy.BASELINE: _baseline,
    # "dense" is experiment A (dense RAG); today it is the baseline pipeline. It gets its
    # own builder only if the two diverge.
    RagStrategy.DENSE: _baseline,
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
