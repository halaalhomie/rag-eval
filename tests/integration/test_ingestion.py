import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import EMBEDDING_DIM, Chunk, Document, ParentChunk
from app.db.session import get_session
from app.ingestion.loaders import load_markdown
from app.ingestion.pipeline import IngestionPipeline, document_id_for
from app.main import create_app
from tests.helpers import make_pdf

pytestmark = pytest.mark.integration

BODY = "\n\n".join(
    ["# Probes"]
    + [
        f"## Probe {i}\n\n"
        + " ".join(f"Liveness probe sentence {i}.{j} restarts containers." for j in range(12))
        for i in range(6)
    ]
)


class FakeEmbedder:
    """Deterministic hash-based vectors; enough to test storage, not semantics."""

    model_name = "fake-hash-embedder"

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(EMBEDDING_DIM)])
        return out


@pytest.fixture
def session(pg_engine):
    with Session(pg_engine) as s:
        s.execute(delete(Document))
        s.commit()
        yield s


def small_chunks() -> Settings:
    return Settings(
        chunking={"child_chunk_size": 80, "child_chunk_overlap": 10, "parent_chunk_size": 300}
    )


def test_ingest_persists_documents_parents_and_chunks_with_valid_offsets(session):
    doc = load_markdown(BODY, source="test://probes.md")
    result = IngestionPipeline(small_chunks()).ingest_document(session, doc)
    session.commit()

    assert result.status == "created" and result.parents > 1 and result.chunks > result.parents
    stored = session.get(Document, result.document_id)
    assert stored.content == doc.text
    assert stored.metadata_["chunking"]["child_chunk_size"] == 80

    chunks = session.scalars(select(Chunk).order_by(Chunk.ordinal)).all()
    parents = {p.id: p for p in session.scalars(select(ParentChunk))}
    assert len(chunks) == result.chunks
    for c in chunks:
        assert stored.content[c.start_char : c.end_char] == c.text
        p = parents[c.parent_id]
        assert p.start_char <= c.start_char and c.end_char <= p.end_char
        assert c.section.startswith("Probes")
        assert c.id == f"{result.document_id}:c{c.ordinal}"


def test_reingest_is_idempotent_and_config_change_rechunks(session):
    doc = load_markdown(BODY, source="test://probes.md")
    first = IngestionPipeline(small_chunks()).ingest_document(session, doc)
    session.commit()
    again = IngestionPipeline(small_chunks()).ingest_document(session, doc)
    assert again.status == "unchanged"

    bigger = Settings(
        chunking={"child_chunk_size": 200, "child_chunk_overlap": 10, "parent_chunk_size": 600}
    )
    changed = IngestionPipeline(bigger).ingest_document(session, doc)
    session.commit()
    assert changed.status == "updated" and changed.chunks < first.chunks
    n = session.query(Chunk).filter_by(document_id=doc_id_of(doc)).count()
    assert n == changed.chunks  # old chunks fully replaced, none orphaned


def doc_id_of(doc):
    return document_id_for(doc.source)


def test_embedder_vectors_are_stored_and_trigger_reingest(session):
    doc = load_markdown(BODY, source="test://probes.md")
    IngestionPipeline(small_chunks()).ingest_document(session, doc)
    session.commit()
    embedder = FakeEmbedder()
    result = IngestionPipeline(small_chunks(), embedder).ingest_document(session, doc)
    session.commit()
    assert result.status == "updated" and embedder.calls == 1
    chunk = session.scalars(select(Chunk).limit(1)).one()
    assert chunk.embedding_model == "fake-hash-embedder"
    assert len(chunk.embedding) == EMBEDDING_DIM


def test_batch_ingest_isolates_failures(session, tmp_path):
    good = tmp_path / "good.md"
    good.write_text(BODY)
    empty = tmp_path / "empty.md"
    empty.write_text("<!-- only a comment -->")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    summary = IngestionPipeline(small_chunks()).ingest_files(
        session, [(good, "t/good.md"), (empty, "t/empty.md"), (bad, "t/bad.pdf")]
    )
    assert [r.status for r in summary.results] == ["created", "skipped", "failed"]
    assert session.query(Document).count() == 1


def test_ingest_endpoint_accepts_uploads_and_reports_per_file_status(session, pg_engine):
    app = create_app()

    def session_override():
        with Session(pg_engine) as s:
            yield s
            s.commit()

    app.dependency_overrides[get_session] = session_override
    client = TestClient(app)
    files = [
        ("files", ("guide.md", BODY.encode(), "text/markdown")),
        (
            "files",
            ("manual.pdf", make_pdf([["Kubelet page one."], ["Page two."]]), "application/pdf"),
        ),
        ("files", ("../../etc/evil.docx", b"x", "application/octet-stream")),
    ]
    resp = client.post("/documents/ingest", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_source = {r["source"]: r for r in body["results"]}
    assert by_source["upload://guide.md"]["status"] == "created"
    assert by_source["upload://manual.pdf"]["status"] == "created"
    assert by_source["upload://evil.docx"]["status"] == "failed"  # path stripped, type rejected
    assert body["summary"]["failed"] == 1

    listed = client.get("/documents").json()
    assert {d["source"] for d in listed} == {"upload://guide.md", "upload://manual.pdf"}
    pdf = next(d for d in listed if d["source_type"] == "pdf")
    assert pdf["chunks"] >= 1
