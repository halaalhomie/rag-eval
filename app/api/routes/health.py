from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_session
from app.models.health import ComponentStatus, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

try:
    APP_VERSION = version("rag-forge")
except PackageNotFoundError:  # running from a source checkout without install
    APP_VERSION = "0.0.0+local"


def _check_database(session: Session) -> ComponentStatus:
    try:
        session.execute(text("SELECT 1"))
        has_vector = session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return ComponentStatus(status="error", detail=type(exc).__name__)
    if not has_vector:
        return ComponentStatus(status="error", detail="pgvector extension not installed")
    return ComponentStatus(status="ok")


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    components = {"database": _check_database(session)}
    healthy = all(c.status == "ok" for c in components.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=APP_VERSION,
        environment=settings.environment.value,
        rag_strategy=settings.retrieval.rag_strategy.value,
        components=components,
    )
