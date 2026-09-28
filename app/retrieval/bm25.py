"""BM25 lexical retrieval (Okapi BM25, in-process inverted index).

score(q, d) = Σ_{t in q} idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (Lucene form; never negative)

Why not PostgreSQL full-text search: `ts_rank` is not BM25 (no IDF saturation, different
length normalization), so a "BM25" experiment on top of it would be mislabelled.

The index is built from the `chunks` table on first use and rebuilt automatically when the
table changes (checked with a cheap signature query), so re-ingestion never leaves it
stale. At ~4k chunks the whole index lives comfortably in memory.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select

from app.config import Settings
from app.db.models import Chunk
from app.db.session import session_scope
from app.ingestion.pipeline import embedding_input
from app.retrieval.base import MetadataFilter, RetrievalResult
from app.retrieval.dense import SessionFactory
from app.retrieval.text import Analyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Doc:
    chunk_id: str
    document_id: str
    parent_id: str | None
    text: str
    section: str | None
    page: int | None
    start_char: int
    end_char: int
    metadata: dict


class BM25Index:
    def __init__(
        self, docs: list[_Doc], analyzer: Analyzer, include_context: bool, k1: float, b: float
    ):
        self.docs = docs
        self.analyzer = analyzer
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, d in enumerate(docs):
            text = (
                embedding_input(d.metadata.get("title"), d.section, d.text)
                if include_context
                else d.text
            )
            tf = Counter(analyzer(text))
            self.lengths.append(sum(tf.values()))
            for term, n in tf.items():
                self.postings[term].append((i, n))
        self.n = len(docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl or 1.0
        for term in set(self.analyzer(query)):  # repeated query terms do not add weight
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = k1 * (1 - b + b * self.lengths[i] / avgdl)
                scores[i] += idf * tf * (k1 + 1) / (tf + norm)
        # Ties broken by chunk id so rankings are deterministic.
        ranked = sorted(scores.items(), key=lambda x: (-x[1], self.docs[x[0]].chunk_id))
        return ranked[:k]


class BM25Retriever:
    name = "bm25"

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        include_context: bool = True,
        session_factory: SessionFactory = session_scope,
    ):
        self.analyzer = analyzer or Analyzer()
        self.k1, self.b = k1, b
        self.include_context = include_context
        self.session_factory = session_factory
        self._index: BM25Index | None = None
        self._signature: tuple | None = None
        self._lock = threading.Lock()

    def _current_signature(self, session) -> tuple:
        return tuple(
            session.execute(select(func.count(Chunk.id), func.max(Chunk.created_at))).one()
        )

    def _ensure_index(self) -> BM25Index:
        with self.session_factory() as session:
            sig = self._current_signature(session)
            if self._index is not None and sig == self._signature:
                return self._index
            with self._lock:
                if self._index is not None and sig == self._signature:
                    return self._index
                started = time.perf_counter()
                rows = session.execute(select(Chunk).order_by(Chunk.id)).scalars()
                docs = [
                    _Doc(
                        c.id,
                        c.document_id,
                        c.parent_id,
                        c.text,
                        c.section,
                        c.page,
                        c.start_char,
                        c.end_char,
                        dict(c.metadata_ or {}),
                    )
                    for c in rows
                ]
                self._index = BM25Index(docs, self.analyzer, self.include_context, self.k1, self.b)
                self._signature = sig
                logger.info(
                    "Built BM25 index: %d chunks, %d terms, %s, %.2fs",
                    len(docs),
                    len(self._index.idf),
                    self.analyzer.name(),
                    time.perf_counter() - started,
                )
                return self._index

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievalResult]:
        if k <= 0 or not query.strip():
            return []
        index = self._ensure_index()
        # Filters are applied after scoring, so search deeper when filtering.
        depth = index.n if filters else k
        results = []
        for i, score in index.search(query, depth):
            d = index.docs[i]
            if filters and not all(d.metadata.get(key) == v for key, v in filters.items()):
                continue
            results.append(
                RetrievalResult(
                    document_id=d.document_id,
                    chunk_id=d.chunk_id,
                    text=d.text,
                    score=score,
                    rank=len(results) + 1,
                    retriever=self.name,
                    parent_id=d.parent_id,
                    section=d.section,
                    page=d.page,
                    start_char=d.start_char,
                    end_char=d.end_char,
                    metadata=d.metadata,
                )
            )
            if len(results) == k:
                break
        return results


_SHARED: dict[tuple, BM25Retriever] = {}


def build_bm25(
    settings: Settings, session_factory: SessionFactory = session_scope
) -> BM25Retriever:
    """One retriever (and in-memory index) per configuration and session factory, shared
    across requests so the index is built once rather than on every query."""
    r = settings.retrieval
    analyzer = Analyzer(
        compounds=r.bm25_compounds, camel_parts=r.bm25_camel_parts, stemming=r.bm25_stemming
    )
    key = (analyzer, r.bm25_k1, r.bm25_b, r.bm25_include_context, id(session_factory))
    if key not in _SHARED:
        _SHARED[key] = BM25Retriever(
            analyzer,
            k1=r.bm25_k1,
            b=r.bm25_b,
            include_context=r.bm25_include_context,
            session_factory=session_factory,
        )
    return _SHARED[key]
