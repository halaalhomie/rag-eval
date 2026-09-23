from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.api.routes.query import get_components
from app.config import Settings, SimilarityMetric
from app.db.models import Document
from app.generation.fake import FakeLLM
from app.ingestion.loaders import load_markdown
from app.ingestion.pipeline import IngestionPipeline
from app.main import create_app
from app.rag.strategies import Components
from app.retrieval.dense import DenseRetriever
from tests.helpers import KeywordEmbedder

pytestmark = pytest.mark.integration

DOCS = {
    "probes.md": ("Probes", "## Liveness\n\nA liveness probe restarts containers. probe probe."),
    "deploy.md": ("Deployments", "## Rollback\n\nA deployment rollback restores a revision."),
    "secrets.md": ("Secrets", "## Secret data\n\nA secret stores sensitive data."),
}


@pytest.fixture
def factory(pg_engine):
    @contextmanager
    def scope():
        with Session(pg_engine) as s:
            yield s
            s.commit()

    return scope


@pytest.fixture
def indexed(pg_engine, factory):
    """Three tiny documents embedded with the keyword embedder."""
    embedder = KeywordEmbedder()
    with Session(pg_engine) as s:
        s.execute(delete(Document))
        s.commit()
        pipeline = IngestionPipeline(Settings(), embedder)
        for name, (title, body) in DOCS.items():
            doc = load_markdown(f"---\ntitle: {title}\n---\n{body}\n", source=f"test://{name}")
            doc.metadata["doc_category"] = "tasks" if name == "deploy.md" else "concepts"
            pipeline.ingest_document(s, doc)
        s.commit()
    return embedder


def test_dense_ranks_by_similarity_and_returns_standard_results(indexed, factory):
    results = DenseRetriever(indexed, session_factory=factory).retrieve("liveness probe", 3)
    top = results[0]
    assert top.metadata["title"] == "Probes"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score
    assert -1.0 <= top.score <= 1.0 + 1e-6
    assert top.retriever == "dense" and top.parent_id and top.end_char > top.start_char


@pytest.mark.parametrize("metric", list(SimilarityMetric))
def test_all_metrics_agree_on_top_result_for_normalized_vectors(indexed, factory, metric):
    retriever = DenseRetriever(indexed, metric=metric, session_factory=factory)
    assert retriever.retrieve("secret", 1)[0].metadata["title"] == "Secrets"


def test_metadata_filter_restricts_results(indexed, factory):
    results = DenseRetriever(indexed, session_factory=factory).retrieve(
        "liveness probe", 5, filters={"doc_category": "tasks"}
    )
    assert [r.metadata["title"] for r in results] == ["Deployments"]


def test_vectors_from_other_embedding_models_are_ignored(indexed, factory):
    other = KeywordEmbedder(model_name="some-other-model")
    assert DenseRetriever(other, session_factory=factory).retrieve("probe", 5) == []


def test_empty_query_or_zero_k_returns_nothing(indexed, factory):
    retriever = DenseRetriever(indexed, session_factory=factory)
    assert retriever.retrieve("   ", 5) == [] and retriever.retrieve("probe", 0) == []


def test_query_endpoint_end_to_end_with_fake_llm(indexed, factory):
    llm = FakeLLM(replies=["A liveness probe restarts containers [1]."])
    app = create_app()
    app.dependency_overrides[get_components] = lambda: Components(llm, indexed, factory)
    client = TestClient(app)

    resp = client.post("/query", json={"query": "what does a liveness probe do", "top_k": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["retrieval_strategy"] == "baseline" and body["retrieval_iterations"] == 1
    assert body["citations"][0]["chunk_id"].endswith(":c0")
    assert body["citations"][0]["title"] == "Probes"
    assert body["sources"] == ["[1] Probes > Liveness"]
    assert body["contexts"][0]["text"] is None  # text only on request
    assert body["usage"]["llm_calls"] == 1 and body["trace_id"]


def test_query_endpoint_rejects_unimplemented_strategy(indexed, factory):
    app = create_app()
    app.dependency_overrides[get_components] = lambda: Components(FakeLLM(), indexed, factory)
    resp = TestClient(app).post("/query", json={"query": "x", "strategy": "self_rag"})
    assert resp.status_code == 501
    assert "baseline" in resp.json()["detail"]
