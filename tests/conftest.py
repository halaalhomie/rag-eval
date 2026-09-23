"""Shared fixtures.

Must not import `app` at module level before the environment is pinned: settings are read
at import time by `app.db.models` (embedding dimension).
"""

import os
from collections.abc import Iterator

import pytest

# Never let a developer's local .env leak into the test run.
os.environ["RAG_FORGE_ENV_FILE"] = ""
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LLM_PROVIDER", "fake")

# Integration tests use a dedicated database so they never touch development data.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://ragforge:ragforge@localhost:5434/ragforge_test",
)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def pg_engine():
    """Engine bound to a freshly initialized test database; skips if Postgres is down."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.exc import OperationalError

    from app.db.init_db import init_db

    url = make_url(TEST_DATABASE_URL)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable for integration tests: {exc.orig}")
    finally:
        admin.dispose()

    engine = create_engine(url)
    init_db(engine, drop=True)
    yield engine
    engine.dispose()
