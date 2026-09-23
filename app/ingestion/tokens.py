"""Approximate token counting.

Chunk sizes are configured in tokens, but tying ingestion to one model's tokenizer would
make chunk boundaries move whenever the embedding or LLM model changes. Instead this is a
model-independent heuristic, calibrated against a real subword tokenizer.

Calibration (scripts/audit_chunk_tokens.py, bge-small WordPiece, 3,062 Kubernetes-docs
chunks): a naive "one token per word" count gave estimate/actual = 0.87 median and 0.43
worst case. The worst cases were hex IDs, wide tables and long CamelCase identifiers, all
of which WordPiece splits into many pieces. The rules below give a median of 0.98 and a
worst underestimate of 0.70 (see docs/ingestion.md for the measured numbers):

- a word piece costs 1 token per ~7 characters
- a piece mixing letters and digits (hashes, IDs, versions) costs 1 token per ~2 characters
- CamelCase identifiers are costed per component
"""

from __future__ import annotations

import re

_PIECE = re.compile(r"\w+|[^\w\s]")
_CAMEL_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_WORD_CHARS = 7
_MIXED_CHARS = 2


def _cost(run: str, chars_per_token: int) -> int:
    return 1 + (len(run) - 1) // chars_per_token


def piece_tokens(piece: str) -> int:
    if not (piece[0].isalnum() or piece[0] == "_"):
        return 1  # punctuation
    if any(c.isdigit() for c in piece) and any(c.isalpha() for c in piece):
        return _cost(piece, _MIXED_CHARS)
    parts = _CAMEL_PART.findall(piece)
    if len(parts) > 1:
        return sum(_cost(p, _WORD_CHARS) for p in parts)
    return _cost(piece, _WORD_CHARS)


def estimate_tokens(text: str) -> int:
    return sum(piece_tokens(p) for p in _PIECE.findall(text))
