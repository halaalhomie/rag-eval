"""Derive chunk-level relevance from evidence spans against the current index.

A chunk is relevant to an evidence span when it contains at least `min_coverage` of the
span's characters (default 0.5). With overlapping chunks, one span can be covered by more
than one chunk, and every such chunk counts as relevant.

This lets the same dataset evaluate any chunking configuration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.evaluation.datasets.schema import EvalItem, EvidenceSpan


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    chunk_id: str
    document_id: str
    start_char: int
    end_char: int


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def relevant_chunks_for_span(
    span: EvidenceSpan, chunks: list[ChunkSpan], min_coverage: float = 0.5
) -> list[str]:
    length = span.end_char - span.start_char
    return [
        c.chunk_id
        for c in chunks
        if c.document_id == span.document_id
        and overlap(span.start_char, span.end_char, c.start_char, c.end_char)
        >= min_coverage * length
    ]


class ChunkIndex:
    """In-memory map document_id -> chunk spans, loaded once per evaluation run."""

    def __init__(self, by_document: dict[str, list[ChunkSpan]]):
        self.by_document = by_document

    @classmethod
    def load(cls, session: Session, document_ids: set[str] | None = None) -> ChunkIndex:
        stmt = select(Chunk.id, Chunk.document_id, Chunk.start_char, Chunk.end_char)
        if document_ids is not None:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        by_doc: dict[str, list[ChunkSpan]] = defaultdict(list)
        for row in session.execute(stmt):
            by_doc[row.document_id].append(ChunkSpan(*row))
        return cls(dict(by_doc))

    def evidence_chunks(self, item: EvalItem, min_coverage: float = 0.5) -> list[set[str]]:
        """One set of relevant chunk IDs per evidence span (a span is 'found' if any of its
        chunks is retrieved)."""
        return [
            set(
                relevant_chunks_for_span(
                    span, self.by_document.get(span.document_id, []), min_coverage
                )
            )
            for span in item.evidence
        ]

    def relevant_chunks(self, item: EvalItem, min_coverage: float = 0.5) -> set[str]:
        out: set[str] = set()
        for s in self.evidence_chunks(item, min_coverage):
            out |= s
        return out
