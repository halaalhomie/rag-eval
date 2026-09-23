from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import RagStrategy, Settings, get_settings
from app.generation.citations import Citation
from app.generation.factory import build_llm
from app.generation.llm import LLMError
from app.generation.usage import UsageSummary
from app.rag.strategies import Components, StrategyNotImplementedError, build_pipeline
from app.rag.types import Timings
from app.retrieval.embeddings import EmbeddingError, get_embedder

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    strategy: RagStrategy | None = Field(default=None, description="defaults to RAG_STRATEGY")
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = Field(
        default=None, description='chunk metadata containment, e.g. {"doc_category": "tasks"}'
    )
    include_context_text: bool = False


class ContextOut(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    score: float
    retriever: str
    title: str | None
    section: str | None
    url: str | None
    text: str | None = None


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    sources: list[str]
    invalid_citations: list[int]
    retrieval_strategy: str
    retrieval_iterations: int
    evidence_score: float | None
    faithfulness_score: float | None
    trace_id: str
    contexts: list[ContextOut]
    timings: Timings
    usage: UsageSummary


@lru_cache(maxsize=1)
def _components() -> Components:
    settings = get_settings()
    return Components(llm=build_llm(settings.llm), embedder=get_embedder(settings.embedding))


def get_components() -> Components:
    return _components()


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    settings: Settings = Depends(get_settings),
    components: Components = Depends(get_components),
) -> QueryResponse:
    strategy = req.strategy or settings.retrieval.rag_strategy
    try:
        pipeline = build_pipeline(strategy, settings, components)
        result = pipeline.run(req.query, top_k=req.top_k, filters=req.filters)
    except StrategyNotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    except EmbeddingError as exc:
        logger.exception("Embedding failure")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "embedding model unavailable"
        ) from exc
    except LLMError as exc:
        logger.warning("LLM failure: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"LLM backend error ({type(exc).__name__})"
        ) from exc

    return QueryResponse(
        answer=result.answer,
        abstained=result.abstained,
        citations=result.citations,
        sources=[c.label() for c in result.citations],
        invalid_citations=result.invalid_citations,
        retrieval_strategy=result.strategy,
        retrieval_iterations=result.retrieval_iterations,
        evidence_score=result.evidence_score,
        faithfulness_score=result.faithfulness_score,
        trace_id=result.trace_id,
        contexts=[
            ContextOut(
                rank=c.rank,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                score=round(c.score, 6),
                retriever=c.retriever,
                title=c.title,
                section=c.section,
                url=c.metadata.get("url"),
                text=c.text if req.include_context_text else None,
            )
            for c in result.contexts
        ],
        timings=result.timings,
        usage=result.usage,
    )
