"""Data types shared by loaders, chunkers and the ingestion pipeline.

All spans are half-open character offsets `[start, end)` into `LoadedDocument.text`.
Chunkers never rewrite text: every parent and child chunk is an exact slice of the
normalized document text. That invariant is what lets evaluation labels be anchored to
spans instead of chunk IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Section:
    """A structural region of a document (a Markdown heading section or a PDF page)."""

    start: int
    end: int
    heading_path: tuple[str, ...] = ()  # e.g. ("Pod lifetime", "Pods and fault recovery")
    page: int | None = None

    @property
    def label(self) -> str | None:
        return " > ".join(self.heading_path) or None


@dataclass(slots=True)
class LoadedDocument:
    source: str  # stable identifier: corpus-relative path or upload://name
    name: str  # human-readable title
    source_type: str  # markdown | txt | pdf
    text: str  # normalized text; all offsets point into this
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    tokens: int


@dataclass(frozen=True, slots=True)
class ParentSpan:
    start: int
    end: int
    tokens: int
    section: str | None
    page: int | None
    section_paths: tuple[str, ...]  # labels of every section merged into this parent


@dataclass(frozen=True, slots=True)
class ChildSpan:
    start: int
    end: int
    tokens: int
    parent_index: int
