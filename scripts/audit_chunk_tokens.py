"""Compare the chunker's token estimate with the embedding model's real tokenizer.

Reports the estimate/actual ratio and how many embedding inputs exceed the model's
max_seq_length (i.e. would be silently truncated before embedding).

Usage:
    python scripts/audit_chunk_tokens.py
"""

import json
import statistics

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Chunk
from app.db.session import session_scope
from app.ingestion.pipeline import embedding_input
from app.retrieval.embeddings import get_embedder

if __name__ == "__main__":
    settings = get_settings()
    embedder = get_embedder(settings.embedding)
    use_ctx = settings.embedding.embedding_include_context
    with session_scope() as s:
        rows = s.execute(
            select(Chunk.text, Chunk.token_count, Chunk.section, Chunk.metadata_)
        ).all()

    ratios, over_limit, actual_counts = [], 0, []
    limit = embedder.max_seq_length
    for text, estimate, section, meta in rows:
        actual = embedder.token_count(text)
        ratios.append(estimate / actual)
        inp = embedding_input(meta.get("title"), section, text) if use_ctx else text
        n = embedder.token_count(inp)
        actual_counts.append(n)
        over_limit += n > limit

    q = statistics.quantiles(ratios, n=20)
    print(
        json.dumps(
            {
                "chunks": len(rows),
                "model": embedder.model_name,
                "max_seq_length": limit,
                "estimate_over_actual": {
                    "median": round(statistics.median(ratios), 3),
                    "p5": round(q[0], 3),
                    "p95": round(q[-1], 3),
                },
                "embedding_input_tokens": {
                    "median": statistics.median(actual_counts),
                    "max": max(actual_counts),
                },
                "embedding_inputs_truncated": over_limit,
            },
            indent=2,
        )
    )
