"""File loaders: turn a file into a `LoadedDocument` with normalized text and sections."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.markdown import (
    Preprocessor,
    markdown_sections,
    normalize_markdown,
    split_front_matter,
)
from app.ingestion.types import LoadedDocument, Section


class IngestionError(Exception):
    """Base class for errors that should skip one file without aborting a batch."""


class UnsupportedFileTypeError(IngestionError):
    pass


class EmptyDocumentError(IngestionError):
    pass


class DocumentParseError(IngestionError):
    pass


SUFFIX_TYPES = {".md": "markdown", ".markdown": "markdown", ".txt": "txt", ".pdf": "pdf"}
_BLANK_RUN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _normalize_plain(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def load_markdown(
    raw: str, source: str, name: str | None = None, preprocessor: Preprocessor | None = None
) -> LoadedDocument:
    front_matter, body = split_front_matter(raw)
    if preprocessor is not None:
        body = preprocessor(body)
    text = normalize_markdown(body)
    title = str(front_matter.get("title") or name or Path(source).stem)
    metadata: dict[str, Any] = {"title": title}
    for key in ("description", "content_type", "weight"):
        if key in front_matter:
            metadata[key] = front_matter[key]
    return LoadedDocument(
        source=source,
        name=title,
        source_type="markdown",
        text=text,
        sections=markdown_sections(text),
        metadata=metadata,
    )


def load_text(raw: str, source: str, name: str | None = None) -> LoadedDocument:
    text = _normalize_plain(raw)
    text = text + "\n" if text else ""
    title = name or Path(source).stem
    return LoadedDocument(
        source=source,
        name=title,
        source_type="txt",
        text=text,
        sections=[Section(0, len(text))] if text else [],
        metadata={"title": title},
    )


def load_pdf(path: Path, source: str, name: str | None = None) -> LoadedDocument:
    """One section per page, so every chunk can be cited with a page number.

    Heading detection on PDF text is unreliable, so pages are the structural unit.
    """
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError, ValueError) as exc:
        raise DocumentParseError(f"{source}: cannot parse PDF ({exc})") from exc

    parts: list[str] = []
    sections: list[Section] = []
    offset = 0
    for page_no, page_text in enumerate(pages, start=1):
        cleaned = _normalize_plain(page_text)
        if not cleaned:
            continue
        if parts:
            parts.append("\n\n")
            offset += 2
        parts.append(cleaned)
        sections.append(Section(offset, offset + len(cleaned), page=page_no))
        offset += len(cleaned)
    text = "".join(parts)
    meta_title = (reader.metadata.title if reader.metadata else None) or None
    title = name or meta_title or Path(source).stem
    return LoadedDocument(
        source=source,
        name=title,
        source_type="pdf",
        text=text,
        sections=sections,
        metadata={"title": title, "page_count": len(pages)},
    )


def load_document(
    path: Path,
    source: str | None = None,
    name: str | None = None,
    preprocessor: Preprocessor | None = None,
) -> LoadedDocument:
    """Dispatch on file suffix. `preprocessor` applies to Markdown only."""
    source = source or str(path)
    source_type = SUFFIX_TYPES.get(path.suffix.lower())
    if source_type is None:
        raise UnsupportedFileTypeError(
            f"{source}: unsupported file type '{path.suffix}' "
            f"(supported: {', '.join(sorted(SUFFIX_TYPES))})"
        )
    if source_type == "pdf":
        doc = load_pdf(path, source, name)
    else:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(f"{source}: not valid UTF-8 text") from exc
        doc = (
            load_markdown(raw, source, name, preprocessor)
            if source_type == "markdown"
            else load_text(raw, source, name)
        )
    if not doc.text.strip():
        raise EmptyDocumentError(f"{source}: no extractable text")
    return doc
