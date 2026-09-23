"""Ingestion pipeline: LoadedDocument -> documents / parent_chunks / chunks rows.

Re-ingestion is idempotent. A document is rebuilt only when its *fingerprint* changes:
content hash + chunker configuration + embedding model. Otherwise it is reported as
"unchanged" and left untouched. Changing CHILD_CHUNK_SIZE and re-running ingestion
therefore re-chunks everything, while re-running with the same config is a no-op.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Chunk, Document, ParentChunk
from app.ingestion.chunking import StructureAwareChunker, section_at
from app.ingestion.loaders import EmptyDocumentError, IngestionError, load_document
from app.ingestion.markdown import Preprocessor
from app.ingestion.types import LoadedDocument

logger = logging.getLogger(__name__)


class DocumentEmbedder(Protocol):
    """Implemented by the embedding layer (phase 3). Ingestion only needs this much."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


def document_id_for(source: str) -> str:
    return "doc_" + hashlib.sha256(source.strip().encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(slots=True)
class IngestResult:
    source: str
    document_id: str | None
    status: Literal["created", "updated", "unchanged", "skipped", "failed"]
    parents: int = 0
    chunks: int = 0
    error: str | None = None


@dataclass(slots=True)
class IngestSummary:
    results: list[IngestResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    def count(self, status: str) -> int:
        return sum(r.status == status for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": len(self.results),
            **{s: self.count(s) for s in ("created", "updated", "unchanged", "skipped", "failed")},
            "chunks_written": sum(r.chunks for r in self.results),
            "elapsed_s": round(self.elapsed_s, 2),
        }


class IngestionPipeline:
    def __init__(self, settings: Settings, embedder: DocumentEmbedder | None = None):
        self.settings = settings
        self.chunker = StructureAwareChunker(settings.chunking)
        self.embedder = embedder

    def _fingerprint(self, text_hash: str) -> str:
        model = self.embedder.model_name if self.embedder else "none"
        raw = f"{text_hash}|{self.chunker.fingerprint()}|{model}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def ingest_document(self, session: Session, doc: LoadedDocument) -> IngestResult:
        doc_id = document_id_for(doc.source)
        text_hash = content_hash(doc.text)
        fingerprint = self._fingerprint(text_hash)

        existing = session.get(Document, doc_id)
        if existing is not None and existing.metadata_.get("fingerprint") == fingerprint:
            return IngestResult(doc.source, doc_id, "unchanged")

        parents, children = self.chunker.chunk(doc)
        embeddings: list[list[float]] | None = None
        if self.embedder is not None and children:
            embeddings = self.embedder.embed_documents(
                [doc.text[c.start : c.end] for c in children]
            )

        metadata = {
            **doc.metadata,
            "fingerprint": fingerprint,
            "chunker": self.chunker.fingerprint(),
            "chunking": self.settings.chunking.model_dump(mode="json"),
        }
        if existing is None:
            record = Document(id=doc_id, source=doc.source)
            session.add(record)
            status: Literal["created", "updated"] = "created"
        else:
            record = existing
            # Bulk deletes: ORM cascades would load every child row first.
            session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            session.execute(delete(ParentChunk).where(ParentChunk.document_id == doc_id))
            status = "updated"
        record.name = doc.name
        record.source_type = doc.source_type
        record.content_hash = text_hash
        record.char_count = len(doc.text)
        record.content = doc.text
        record.metadata_ = metadata
        session.flush()

        parent_ids = [f"{doc_id}:p{i}" for i in range(len(parents))]
        session.add_all(
            ParentChunk(
                id=parent_ids[i],
                document_id=doc_id,
                ordinal=i,
                section=p.section,
                page=p.page,
                start_char=p.start,
                end_char=p.end,
                token_count=p.tokens,
                text=doc.text[p.start : p.end],
                metadata_={"sections": list(p.section_paths)},
            )
            for i, p in enumerate(parents)
        )
        session.flush()

        base_meta = {k: doc.metadata[k] for k in ("title", "url") if k in doc.metadata}
        rows = []
        for j, c in enumerate(children):
            sec = section_at(doc.sections, c.start)
            rows.append(
                Chunk(
                    id=f"{doc_id}:c{j}",
                    document_id=doc_id,
                    parent_id=parent_ids[c.parent_index],
                    ordinal=j,
                    section=(sec.label if sec else None) or parents[c.parent_index].section,
                    page=sec.page if sec else parents[c.parent_index].page,
                    start_char=c.start,
                    end_char=c.end,
                    token_count=c.tokens,
                    text=doc.text[c.start : c.end],
                    embedding=embeddings[j] if embeddings else None,
                    embedding_model=self.embedder.model_name if embeddings else None,
                    metadata_={**base_meta, "source_type": doc.source_type},
                )
            )
        session.add_all(rows)
        session.flush()
        return IngestResult(doc.source, doc_id, status, len(parents), len(children))

    def ingest_files(
        self,
        session: Session,
        files: Iterable[tuple[Path, str]],
        preprocessor: Preprocessor | None = None,
        metadata_for: Callable[[Path, LoadedDocument], dict[str, Any]] | None = None,
    ) -> IngestSummary:
        """Ingest (path, source) pairs. Each file commits independently, so one bad file
        never rolls back the rest of the batch.

        `metadata_for(path, doc) -> dict` attaches corpus-specific metadata (e.g. URL).
        """
        summary = IngestSummary()
        started = time.perf_counter()
        for path, source in files:
            try:
                doc = load_document(path, source=source, preprocessor=preprocessor)
                if metadata_for is not None:
                    doc.metadata.update(metadata_for(path, doc))
                result = self.ingest_document(session, doc)
                session.commit()
            except EmptyDocumentError as exc:
                session.rollback()
                result = IngestResult(source, None, "skipped", error=str(exc))
            except IngestionError as exc:
                session.rollback()
                result = IngestResult(source, None, "failed", error=str(exc))
                logger.warning("Skipped %s", exc)
            except Exception as exc:
                session.rollback()
                logger.exception("Failed to ingest %s", source)
                result = IngestResult(source, None, "failed", error=type(exc).__name__)
            summary.results.append(result)
            if len(summary.results) % 50 == 0:
                logger.info("Ingested %d files", len(summary.results))
        summary.elapsed_s = time.perf_counter() - started
        return summary


def corpus_stats(session: Session) -> dict[str, int]:
    return {
        "documents": session.scalar(select(func.count()).select_from(Document)) or 0,
        "parent_chunks": session.scalar(select(func.count()).select_from(ParentChunk)) or 0,
        "chunks": session.scalar(select(func.count()).select_from(Chunk)) or 0,
        "chunks_embedded": session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
        )
        or 0,
    }
