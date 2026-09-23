"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, health, query
from app.config import get_settings
from app.config.settings import Environment
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.environment is not Environment.TEST:
        # Load embedding weights before serving: otherwise the first request pays ~10 s
        # of model loading, which also distorts latency measurements.
        from app.retrieval.embeddings import EmbeddingError, get_embedder

        try:
            get_embedder(settings.embedding).embed_query("warmup")
        except EmbeddingError:
            logger.exception("Embedding model failed to load; /query will return 503")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.observability.log_level, settings.observability.log_json)
    app = FastAPI(
        title="RAG-Forge",
        description="Evaluation-driven adaptive RAG system",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    return app


app = create_app()
