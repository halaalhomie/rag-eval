"""Structure-aware parent/child chunking.

Invariants (enforced by tests):
- every parent and child is an exact slice of the document text: `text[start:end]`
- every child lies entirely inside its parent
- children never exceed CHILD_CHUNK_SIZE and parents never exceed PARENT_CHUNK_SIZE
  (pieces with no whitespace that are still too large are split by characters)
- output is deterministic for a given text + ChunkingSettings

Algorithm
1. Blocks: paragraphs separated by blank lines. A fenced code block is one block, and a
   heading line is glued to the block that follows it (no heading-only chunks).
2. Units: blocks larger than the target size are split into sentences (lines for code),
   then word windows, then character slices as a last resort.
3. Parents: consecutive heading sections are merged while they fit in PARENT_CHUNK_SIZE
   (never across PDF pages). Oversized sections are split at block boundaries. Parents
   therefore always begin at a section or block boundary. An earlier version also refused
   to merge across top-level (H2) sections, but on the Kubernetes docs that made 50% of
   children identical to their parent, so parent-child retrieval added no context.
4. Children: units are packed greedily up to CHILD_CHUNK_SIZE inside each parent. The next
   child starts with up to CHILD_CHUNK_OVERLAP tokens of trailing context: whole units if
   they fit, otherwise trailing sentences of the last prose unit. Overlap is therefore
   best-effort and always aligned to sentence or paragraph boundaries.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass

from app.config.settings import ChunkingSettings
from app.ingestion.markdown import iter_lines, parse_heading
from app.ingestion.tokens import estimate_tokens
from app.ingestion.types import ChildSpan, LoadedDocument, ParentSpan, Section, Span

CHUNKER_VERSION = "structure-aware-v5"

# Source Markdown is hard-wrapped, so a bare newline is NOT a sentence boundary. Split after
# terminal punctuation, after a line-ending colon, or before a list item.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=:)\n|\n(?=[ \t]*(?:[-*+]|\d+[.)])[ \t])")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    tokens: int
    is_code: bool = False


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _unit(text: str, start: int, end: int, is_code: bool = False) -> _Unit | None:
    start, end = _trim(text, start, end)
    if start >= end:
        return None
    return _Unit(start, end, estimate_tokens(text[start:end]), is_code)


def _blocks(text: str, start: int, end: int) -> list[_Unit]:
    """Paragraph / code-fence blocks within [start, end), headings glued forward."""
    raw: list[tuple[int, int, bool, bool]] = []  # (start, end, is_code, is_heading)
    cur_start: int | None = None
    cur_end = start
    cur_code = False
    region = text[start:end]
    for rel, line, in_code in iter_lines(region):
        line_start = start + rel
        line_end = line_start + len(line)
        if line_end > end:
            line_end = end
        if not line.strip() and not in_code:
            if cur_start is not None:
                raw.append((cur_start, cur_end, cur_code, False))
                cur_start = None
            continue
        heading = not in_code and parse_heading(line) is not None
        if heading:
            if cur_start is not None:
                raw.append((cur_start, cur_end, cur_code, False))
            raw.append((line_start, line_end, False, True))
            cur_start = None
            continue
        if cur_start is None:
            cur_start, cur_code = line_start, in_code
        cur_code = cur_code or in_code
        cur_end = line_end
    if cur_start is not None:
        raw.append((cur_start, cur_end, cur_code, False))

    blocks: list[_Unit] = []
    pending_heading: int | None = None
    for b_start, b_end, is_code, is_heading in raw:
        if is_heading:
            pending_heading = b_start if pending_heading is None else pending_heading
            continue
        if pending_heading is not None:
            b_start, pending_heading = pending_heading, None
        if (u := _unit(text, b_start, b_end, is_code)) is not None:
            blocks.append(u)
    if pending_heading is not None and (u := _unit(text, pending_heading, end)) is not None:
        blocks.append(u)  # trailing heading with no body
    return blocks


def _split_on(text: str, unit: _Unit, pattern: re.Pattern[str]) -> list[_Unit]:
    pieces, pos = [], unit.start
    for m in pattern.finditer(text, unit.start, unit.end):
        if (u := _unit(text, pos, m.start(), unit.is_code)) is not None:
            pieces.append(u)
        pos = m.end()
    if (u := _unit(text, pos, unit.end, unit.is_code)) is not None:
        pieces.append(u)
    return pieces


def _word_windows(text: str, unit: _Unit, max_tokens: int) -> list[_Unit]:
    windows, w_start, w_end, w_tok = [], None, unit.start, 0
    for m in _WORD.finditer(text, unit.start, unit.end):
        tok = estimate_tokens(m.group())
        if tok > max_tokens:  # one whitespace-free run bigger than a chunk: slice it
            if w_start is not None:
                windows.append(_Unit(w_start, w_end, w_tok, unit.is_code))
                w_start, w_tok = None, 0
            windows.extend(_char_slices(text, m.start(), m.end(), max_tokens, unit.is_code))
            continue
        if w_start is not None and w_tok + tok > max_tokens:
            windows.append(_Unit(w_start, w_end, w_tok, unit.is_code))
            w_start, w_tok = None, 0
        if w_start is None:
            w_start = m.start()
        w_end, w_tok = m.end(), w_tok + tok
    if w_start is not None:
        windows.append(_Unit(w_start, w_end, w_tok, unit.is_code))
    return windows


def _char_slices(text: str, start: int, end: int, max_tokens: int, is_code: bool) -> list[_Unit]:
    slices, pos = [], start
    step = max(1, (end - start) * max_tokens // max(estimate_tokens(text[start:end]), 1))
    while pos < end:
        stop = min(pos + step, end)
        while stop > pos + 1 and estimate_tokens(text[pos:stop]) > max_tokens:
            stop -= max(1, (stop - pos) // 10)
        slices.append(_Unit(pos, stop, estimate_tokens(text[pos:stop]), is_code))
        pos = stop
    return slices


def _units(text: str, start: int, end: int, max_tokens: int) -> list[_Unit]:
    out: list[_Unit] = []
    for block in _blocks(text, start, end):
        if block.tokens <= max_tokens:
            out.append(block)
            continue
        splitter = re.compile(r"\n") if block.is_code else _SENTENCE_END
        for piece in _split_on(text, block, splitter):
            if piece.tokens <= max_tokens:
                out.append(piece)
            else:
                out.extend(_word_windows(text, piece, max_tokens))
    return out


def _overlap_tail(text: str, units: list[_Unit], budget: int) -> list[_Unit]:
    """Trailing context for the next chunk: whole units, else trailing sentences."""
    if budget <= 0 or not units:
        return []
    tail: list[_Unit] = []
    used = 0
    for u in reversed(units):
        if used + u.tokens > budget:
            break
        tail.insert(0, u)
        used += u.tokens
    if tail or units[-1].is_code:
        return tail
    last = units[-1]
    sentences = _split_on(text, last, _SENTENCE_END)
    start = None
    for s in reversed(sentences):
        if used + s.tokens > budget:
            break
        start, used = s.start, used + s.tokens
    if start is None or start == last.start:
        return []
    return [_Unit(start, last.end, used)]


def _pack(text: str, units: list[_Unit], max_tokens: int, overlap: int) -> list[Span]:
    spans: list[Span] = []
    cur: list[_Unit] = []
    cur_tokens = 0

    def emit() -> None:
        start, end = cur[0].start, cur[-1].end
        spans.append(Span(start, end, estimate_tokens(text[start:end])))

    for u in units:
        if cur and cur_tokens + u.tokens > max_tokens:
            emit()
            tail = _overlap_tail(text, cur, overlap)
            tail_tokens = sum(t.tokens for t in tail)
            if tail_tokens + u.tokens > max_tokens:
                tail, tail_tokens = [], 0
            cur, cur_tokens = tail, tail_tokens
        cur.append(u)
        cur_tokens += u.tokens
    if cur:
        emit()
    return spans


def section_at(sections: list[Section], offset: int) -> Section | None:
    """Most specific section containing `offset` (sections are sorted by start)."""
    starts = [s.start for s in sections]
    i = bisect.bisect_right(starts, offset) - 1
    return sections[i] if i >= 0 and sections[i].start <= offset < sections[i].end else None


class StructureAwareChunker:
    def __init__(self, settings: ChunkingSettings):
        self.settings = settings

    def fingerprint(self) -> str:
        """Changes whenever chunk boundaries could change; stored per document."""
        payload = {"version": CHUNKER_VERSION, **self.settings.model_dump(mode="json")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def parents(self, doc: LoadedDocument) -> list[ParentSpan]:
        text, limit = doc.text, self.settings.parent_chunk_size
        parents: list[ParentSpan] = []
        group: list[Section] = []
        group_tokens = 0

        def key(s: Section) -> int | None:
            return s.page

        def flush() -> None:
            nonlocal group, group_tokens
            if group:
                start, end = _trim(text, group[0].start, group[-1].end)
                if start < end:
                    parents.append(
                        ParentSpan(
                            start=start,
                            end=end,
                            tokens=estimate_tokens(text[start:end]),
                            section=group[0].label,
                            page=group[0].page,
                            section_paths=tuple(s.label or "" for s in group),
                        )
                    )
            group, group_tokens = [], 0

        for sec in doc.sections:
            tokens = estimate_tokens(text[sec.start : sec.end])
            if tokens > limit:
                flush()
                units = _units(text, sec.start, sec.end, limit)
                for span in _pack(text, units, limit, overlap=0):
                    parents.append(
                        ParentSpan(
                            start=span.start,
                            end=span.end,
                            tokens=span.tokens,
                            section=sec.label,
                            page=sec.page,
                            section_paths=(sec.label or "",),
                        )
                    )
                continue
            if group and (key(sec) != key(group[0]) or group_tokens + tokens > limit):
                flush()
            group.append(sec)
            group_tokens += tokens
        flush()
        return parents

    def children(self, doc: LoadedDocument, parents: list[ParentSpan]) -> list[ChildSpan]:
        s = self.settings
        children: list[ChildSpan] = []
        for i, parent in enumerate(parents):
            units = _units(doc.text, parent.start, parent.end, s.child_chunk_size)
            for span in _pack(doc.text, units, s.child_chunk_size, s.child_chunk_overlap):
                children.append(ChildSpan(span.start, span.end, span.tokens, parent_index=i))
        return children

    def chunk(self, doc: LoadedDocument) -> tuple[list[ParentSpan], list[ChildSpan]]:
        parents = self.parents(doc)
        return parents, self.children(doc, parents)
