import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.db.init_db import init_db
from app.db.models import EMBEDDING_DIM, Chunk, Document, ParentChunk
from app.db.session import get_session
from app.main import create_app

pytestmark = pytest.mark.integration


def test_schema_tables_and_hnsw_index_exist(pg_engine):
    insp = inspect(pg_engine)
    assert {"documents", "parent_chunks", "chunks", "experiments"} <= set(insp.get_table_names())
    index_names = {ix["name"] for ix in insp.get_indexes("chunks")}
    assert "ix_chunks_embedding_hnsw_cosine" in index_names


def test_init_db_is_idempotent(pg_engine):
    init_db(pg_engine)
    init_db(pg_engine)


def test_document_parent_chunk_roundtrip_with_vector_search(pg_engine):
    with Session(pg_engine) as s:
        doc = Document(
            id="doc-test",
            name="t.md",
            source="test://t.md",
            source_type="markdown",
            content_hash="x",
            char_count=20,
            content="hello world parent..",
        )
        parent = ParentChunk(
            id="doc-test:p0",
            document_id=doc.id,
            ordinal=0,
            section="Intro",
            start_char=0,
            end_char=20,
            token_count=5,
            text="hello world parent",
        )
        near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
        s.add_all(
            [
                doc,
                parent,
                Chunk(
                    id="doc-test:c0",
                    document_id=doc.id,
                    parent_id=parent.id,
                    ordinal=0,
                    start_char=0,
                    end_char=10,
                    token_count=2,
                    text="near",
                    embedding=near,
                    metadata_={"lang": "en"},
                ),
                Chunk(
                    id="doc-test:c1",
                    document_id=doc.id,
                    parent_id=parent.id,
                    ordinal=1,
                    start_char=10,
                    end_char=20,
                    token_count=2,
                    text="far",
                    embedding=far,
                ),
            ]
        )
        s.commit()

        ranked = s.scalars(
            Chunk.__table__.select()
            .with_only_columns(Chunk.id)
            .order_by(Chunk.embedding.cosine_distance(near))
        ).all()
        assert ranked == ["doc-test:c0", "doc-test:c1"]
        assert s.get(Chunk, "doc-test:c0").parent.section == "Intro"

        # Deleting a document cascades to its parents and chunks at the DB level.
        s.execute(text("DELETE FROM documents WHERE id = 'doc-test'"))
        s.commit()
        assert s.execute(text("SELECT count(*) FROM chunks")).scalar() == 0


def test_health_endpoint_against_real_database(pg_engine):
    app = create_app()

    def session_override():
        with Session(pg_engine) as s:
            yield s

    app.dependency_overrides[get_session] = session_override
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["components"]["database"]["status"] == "ok"
