"""Engine and session management (sync SQLAlchemy 2.0 + psycopg3).

Sync is a deliberate choice: the heavy work in this system (embedding, reranking, BM25)
is CPU-bound and runs in a threadpool anyway, and sync code keeps the retrieval layer
simple to test and to call from scripts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    db = get_settings().database
    return create_engine(
        db.database_url,
        pool_size=db.db_pool_size,
        pool_pre_ping=True,
        echo=db.db_echo,
    )


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on any exception."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session
