"""Result types shared by every RAG strategy.

All strategies return a `RAGResult`, so the API and the experiment runner treat baseline,
corrective and Self-RAG-inspired pipelines uniformly. Later strategies populate more of
the optional fields (evidence score, faithfulness, decisions).
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.generation.citations import Citation
from app.generation.usage import LLMCallRecord, UsageSummary
from app.retrieval.base import RetrievalResult


class Timings(BaseModel):
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    total_s: float = 0.0


class RAGResult(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str
    strategy: str
    answer: str
    abstained: bool
    citations: list[Citation]
    invalid_citations: list[int] = Field(default_factory=list)
    contexts: list[RetrievalResult]
    retrieval_iterations: int = 1
    evidence_score: float | None = None
    faithfulness_score: float | None = None
    timings: Timings = Field(default_factory=Timings)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
