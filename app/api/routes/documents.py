from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path, PurePath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Chunk, Document
from app.db.session import get_session
from app.ingestion.loaders import SUFFIX_TYPES
from app.ingestion.pipeline import IngestionPipeline, IngestResult

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_FILES = 20


class IngestResultOut(BaseModel):
    source: str
    document_id: str | None
    status: str
    parents: int
    chunks: int
    error: str | None


class IngestResponse(BaseModel):
    results: list[IngestResultOut]
    summary: dict[str, float | int]


class DocumentOut(BaseModel):
    id: str
    name: str
    source: str
    source_type: str
    char_count: int
    chunks: int
    url: str | None


def get_pipeline(settings: Settings = Depends(get_settings)) -> IngestionPipeline:
    return IngestionPipeline(settings)


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    files: list[UploadFile] = File(..., description="Markdown, TXT or PDF files"),
    session: Session = Depends(get_session),
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Upload and ingest documents. Each file succeeds or fails independently; the
    per-file status is reported instead of failing the whole request."""
    if len(files) > MAX_FILES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"max {MAX_FILES} files")

    pre_failed: list[IngestResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        staged: list[tuple[Path, str]] = []
        for i, upload in enumerate(files):
            # Keep only the basename; never trust client-supplied paths.
            name = PurePath(upload.filename or f"upload-{i}").name
            source = f"upload://{name}"
            suffix = Path(name).suffix.lower()
            if suffix not in SUFFIX_TYPES:
                pre_failed.append(
                    IngestResult(source, None, "failed", error=f"unsupported file type '{suffix}'")
                )
                continue
            data = upload.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                pre_failed.append(
                    IngestResult(source, None, "failed", error="file exceeds 20 MB limit")
                )
                continue
            path = Path(tmp) / f"{i}{suffix}"
            path.write_bytes(data)
            staged.append((path, source))
        summary = pipeline.ingest_files(session, staged)

    summary.results = pre_failed + summary.results
    return IngestResponse(
        results=[IngestResultOut(**asdict(r)) for r in summary.results],
        summary=summary.as_dict(),
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(
    limit: int = 100, offset: int = 0, session: Session = Depends(get_session)
) -> list[DocumentOut]:
    limit = max(1, min(limit, 1000))
    chunk_counts = (
        select(Chunk.document_id, func.count().label("n")).group_by(Chunk.document_id).subquery()
    )
    rows = session.execute(
        select(Document, func.coalesce(chunk_counts.c.n, 0))
        .outerjoin(chunk_counts, chunk_counts.c.document_id == Document.id)
        .order_by(Document.source)
        .limit(limit)
        .offset(max(0, offset))
    ).all()
    return [
        DocumentOut(
            id=d.id,
            name=d.name,
            source=d.source,
            source_type=d.source_type,
            char_count=d.char_count,
            chunks=n,
            url=d.metadata_.get("url"),
        )
        for d, n in rows
    ]
