"""Approximate token counting.

Chunk sizes are configured in tokens, but tying ingestion to one model's tokenizer would
make chunk boundaries change whenever the embedding or LLM model changes. Instead we count
word and punctuation pieces as a model-independent proxy. Chunk sizes are therefore
approximate by design, and the metadata records `token_count` using this same estimator.
(How far this proxy is from the embedding model's real tokenizer is measured in phase 3.)
"""

from __future__ import annotations

import re

_PIECE = re.compile(r"\w+|[^\w\s]")
# Subword tokenizers split long runs (identifiers, hashes, base64) into many tokens. Counting
# each run as one token would let such text blow past the embedder's input limit.
_CHARS_PER_LONG_TOKEN = 10


def piece_tokens(piece: str) -> int:
    return 1 + (len(piece) - 1) // _CHARS_PER_LONG_TOKEN


def estimate_tokens(text: str) -> int:
    return sum(piece_tokens(p) for p in _PIECE.findall(text))
