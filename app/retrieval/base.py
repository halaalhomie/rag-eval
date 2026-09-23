"""Retriever interface and the standardized result every retriever returns.

All retrievers (dense, BM25, hybrid, reranked) implement `retrieve(query, k, filters)` and
return `RetrievalResult`s, so pipelines and the evaluation framework never depend on a
specific retrieval implementation.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float = Field(description="retriever-specific; higher is better")
    rank: int = Field(ge=1)
    retriever: str
    parent_id: str | None = None
    section: str | None = None
    page: int | None = None
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)  # title, url, category, ...

    @property
    def title(self) -> str | None:
        return self.metadata.get("title")


# Metadata filter: JSON containment on chunk metadata, e.g. {"doc_category": "tasks"}.
MetadataFilter = dict[str, Any]


class Retriever(Protocol):
    name: str

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievalResult]: ...
