"""Relational schema.

documents ──< parent_chunks ──< chunks

- `chunks` are the retrieval units (child chunks). They carry the embedding.
- `parent_chunks` are larger section-level spans used as generation context when
  parent-child retrieval is enabled.
- Both store character offsets into the source document text. Evaluation labels are
  anchored to offsets rather than chunk IDs, so re-chunking with a different config does
  not invalidate the evaluation dataset (chunk relevance is recomputed by span overlap).

The embedding dimension is fixed by EMBEDDING_DIM at schema-creation time. Changing the
embedding model to one with a different dimension requires re-initializing the index.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings

EMBEDDING_DIM = get_settings().embedding.embedding_dim


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar = {dict[str, Any]: JSONB}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    # Deterministic: derived from the normalized source path, so re-ingesting the same file
    # yields the same ID (idempotent upserts, stable eval labels).
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(Text, unique=True)
    source_type: Mapped[str] = mapped_column(String(16))  # pdf | markdown | txt
    content_hash: Mapped[str] = mapped_column(String(64))
    char_count: Mapped[int] = mapped_column(Integer)
    # Full normalized text. All chunk offsets index into it, and evaluation uses it to
    # resolve evidence spans independently of any particular chunking.
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)

    parents: Mapped[list[ParentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)  # heading path, e.g. "Pods > Lifecycle"
    page: Mapped[int | None] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="parents")
    children: Mapped[list[Chunk]] = relationship(back_populates="parent")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("parent_chunks.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(256))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")
    parent: Mapped[ParentChunk | None] = relationship(back_populates="children")

    __table_args__ = (Index("ix_chunks_metadata", "metadata", postgresql_using="gin"),)


class Experiment(Base):
    """Summary row per experiment run. Per-query results live in data/experiments/<id>/."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    strategy: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(64))
    git_commit: Mapped[str | None] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    generation_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    cost_latency: Mapped[dict[str, Any]] = mapped_column(default=dict)
    results_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
