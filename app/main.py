"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import health
from app.config import get_settings
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.observability.log_level, settings.observability.log_json)
    app = FastAPI(
        title="RAG-Forge",
        description="Evaluation-driven adaptive RAG system",
        version="0.1.0",
    )
    app.include_router(health.router)
    return app


app = create_app()
