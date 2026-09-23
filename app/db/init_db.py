"""Schema initialization: pgvector extension, tables, and the ANN index.

Idempotent — safe to run on every container start.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, text

from app.config import SimilarityMetric, get_settings
from app.db.models import Base
from app.db.session import get_engine

logger = logging.getLogger(__name__)

# pgvector operator classes; the index must match the operator used at query time.
HNSW_OPCLASS = {
    SimilarityMetric.COSINE: "vector_cosine_ops",
    SimilarityMetric.L2: "vector_l2_ops",
    SimilarityMetric.INNER_PRODUCT: "vector_ip_ops",
}


def init_db(engine: Engine | None = None, *, drop: bool = False) -> None:
    engine = engine or get_engine()
    metric = get_settings().retrieval.similarity_metric
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    if drop:
        logger.warning("Dropping all tables")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw_{metric.value} "
                f"ON chunks USING hnsw (embedding {HNSW_OPCLASS[metric]})"
            )
        )
    logger.info("Database initialized (similarity metric: %s)", metric.value)
