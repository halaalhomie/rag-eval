"""Markdown parsing: front matter, normalization and heading-based sections."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import yaml

from app.ingestion.types import Section

Preprocessor = Callable[[str], str]

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Closing hashes only count when separated by whitespace, so "## C#" keeps its "#".
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_HEADING_ANCHOR = re.compile(r"[ \t]*\{#[^}]*\}[ \t]*$")  # "## Title {#anchor}"
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_BLANK_RUN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[match.end() :]


def normalize_markdown(body: str) -> str:
    """Canonical text form. Offsets are computed on the output of this function."""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        if _HEADING.match(line):
            line = _HEADING_ANCHOR.sub("", line)
        lines.append(line)
    text = "\n".join(lines)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip() + "\n" if text.strip() else ""


def iter_lines(text: str):
    """Yield (start_offset, line_without_newline, inside_code_fence)."""
    offset = 0
    fence: str | None = None
    for line in text.split("\n"):
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0] * 3
            yield offset, line, True
        elif fence is not None:
            yield offset, line, True
            if m and m.group(1).startswith(fence):
                fence = None
        else:
            yield offset, line, False
        offset += len(line) + 1


def parse_heading(line: str) -> tuple[int, str] | None:
    m = _HEADING.match(line)
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def markdown_sections(text: str) -> list[Section]:
    """Split text at ATX headings (ignoring headings inside code fences).

    Each section spans from its heading line to the next heading of any level, and carries
    the full heading path (ancestors included) so a chunk can be cited as
    "Pod Lifecycle > Container probes > When should you use a startup probe?".
    """
    starts: list[tuple[int, tuple[str, ...]]] = [(0, ())]
    stack: list[tuple[int, str]] = []
    for offset, line, in_code in iter_lines(text):
        if in_code:
            continue
        heading = parse_heading(line)
        if heading is None:
            continue
        level, title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        starts.append((offset, tuple(t for _, t in stack)))

    sections = []
    for i, (start, path) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        if text[start:end].strip():
            sections.append(Section(start=start, end=end, heading_path=path))
    return sections
