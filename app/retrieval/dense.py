"""Dense retrieval over pgvector.

Two HNSW details that silently change results if ignored:
- `hnsw.ef_search` (default 40) caps how many rows an HNSW scan can return, so asking for
  k=50 candidates would quietly return at most 40. It is raised per query to cover k.
- With a metadata filter, HNSW finds neighbours *then* filters, which can return fewer
  than k rows. pgvector >= 0.8 iterative scans keep searching until k rows pass the filter.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import SimilarityMetric
from app.db.models import Chunk
from app.db.session import session_scope
from app.retrieval.base import MetadataFilter, RetrievalResult
from app.retrieval.embeddings import Embedder

SessionFactory = Callable[[], AbstractContextManager[Session]]


def chunk_to_result(chunk: Chunk, score: float, rank: int, retriever: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        text=chunk.text,
        score=score,
        rank=rank,
        retriever=retriever,
        parent_id=chunk.parent_id,
        section=chunk.section,
        page=chunk.page,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        metadata=dict(chunk.metadata_ or {}),
    )


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        embedder: Embedder,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        session_factory: SessionFactory = session_scope,
    ):
        self.embedder = embedder
        self.metric = metric
        self.session_factory = session_factory

    def _distance(self, vector: list[float]):
        col = Chunk.embedding
        match self.metric:
            case SimilarityMetric.COSINE:
                return col.cosine_distance(vector)
            case SimilarityMetric.L2:
                return col.l2_distance(vector)
            case SimilarityMetric.INNER_PRODUCT:
                return col.max_inner_product(vector)  # pgvector returns the *negative* IP

    def _score(self, distance: float) -> float:
        """Convert distance to a higher-is-better similarity score."""
        if self.metric is SimilarityMetric.COSINE:
            return 1.0 - distance
        return -distance  # l2: negative distance; ip: -(-ip) = ip

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievalResult]:
        if k <= 0 or not query.strip():
            return []
        vector = self.embedder.embed_query(query)
        distance = self._distance(vector).label("distance")
        stmt = (
            select(Chunk, distance)
            .where(Chunk.embedding.is_not(None))
            # Never mix vectors from different embedding models in one ranking.
            .where(Chunk.embedding_model == self.embedder.model_name)
            .order_by(distance)
            .limit(k)
        )
        if filters:
            stmt = stmt.where(Chunk.metadata_.contains(filters))
        with self.session_factory() as session:
            session.execute(text(f"SET LOCAL hnsw.ef_search = {max(40, 2 * int(k))}"))
            if filters:
                session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            # relaxed_order may return rows slightly out of order; re-sort to be safe.
            rows = sorted(session.execute(stmt).all(), key=lambda r: r.distance)
            # Build results while the session is open: ORM rows must not be touched after
            # it closes (their attributes may be expired).
            return [
                chunk_to_result(chunk, self._score(dist), rank, self.name)
                for rank, (chunk, dist) in enumerate(rows, start=1)
            ]
