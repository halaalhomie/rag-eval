"""Numbered-source prompting and citation parsing.

Contexts are shown to the LLM as [1]..[n]. The answer cites them inline ("... [2]" or
"[1, 3]"). Parsing maps each number back to the retrieved chunk. Numbers that don't
correspond to a provided source are kept aside as `invalid_citations` — a cheap,
deterministic signal of citation hallucination that the evaluation framework reports.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.retrieval.base import RetrievalResult

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class Citation(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    title: str | None
    section: str | None
    page: int | None
    url: str | None
    start_char: int
    end_char: int

    def label(self) -> str:
        where = " > ".join(p for p in (self.title, self.section) if p) or self.document_id
        if self.page is not None:
            where += f", page {self.page}"
        return f"[{self.number}] {where}"


def format_sources(contexts: list[RetrievalResult]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        header = " > ".join(p for p in (ctx.title, ctx.section) if p) or ctx.document_id
        if ctx.page is not None:
            header += f" (page {ctx.page})"
        blocks.append(f"[{i}] {header}\n{ctx.text}")
    return "\n\n".join(blocks)


def cited_numbers(answer: str) -> list[int]:
    """Citation numbers in order of first appearance."""
    seen: dict[int, None] = {}
    for match in _CITATION.finditer(answer):
        for part in match.group(1).split(","):
            seen.setdefault(int(part), None)
    return list(seen)


def resolve_citations(
    answer: str, contexts: list[RetrievalResult]
) -> tuple[list[Citation], list[int]]:
    """Returns (valid citations in first-cited order, invalid citation numbers)."""
    valid, invalid = [], []
    for n in cited_numbers(answer):
        if 1 <= n <= len(contexts):
            ctx = contexts[n - 1]
            valid.append(
                Citation(
                    number=n,
                    chunk_id=ctx.chunk_id,
                    document_id=ctx.document_id,
                    title=ctx.title,
                    section=ctx.section,
                    page=ctx.page,
                    url=ctx.metadata.get("url"),
                    start_char=ctx.start_char,
                    end_char=ctx.end_char,
                )
            )
        else:
            invalid.append(n)
    return valid, invalid
