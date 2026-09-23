"""/health behavior with the database dependency stubbed out."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_session
from app.main import create_app


def _client_with_session(session) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    return TestClient(app)


def test_health_ok_when_db_and_pgvector_available():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = 1
    resp = _client_with_session(session).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["components"]["database"]["status"] == "ok"
    assert body["rag_strategy"] == "adaptive"


def test_health_degraded_when_pgvector_missing():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = None
    resp = _client_with_session(session).get("/health")
    assert resp.status_code == 503
    assert resp.json()["components"]["database"]["detail"] == "pgvector extension not installed"


def test_health_degraded_and_no_error_leak_when_db_down():
    session = MagicMock()
    session.execute.side_effect = OperationalError("SELECT 1", {}, Exception("password=x"))
    resp = _client_with_session(session).get("/health")
    assert resp.status_code == 503
    detail = resp.json()["components"]["database"]["detail"]
    assert detail == "OperationalError"  # exception type only; no connection details
