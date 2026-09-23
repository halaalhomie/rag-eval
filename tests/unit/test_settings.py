import pytest
from pydantic import ValidationError

from app.config import FusionMethod, LLMProvider, RagStrategy, Settings
from app.config.settings import (
    AdaptiveSettings,
    ChunkingSettings,
    DatabaseSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
)


def test_defaults_are_valid_and_match_documented_values():
    s = Settings()
    assert s.retrieval.rag_strategy is RagStrategy.ADAPTIVE
    assert s.retrieval.top_k == 10
    assert s.retrieval.rerank_top_k == 5
    assert s.retrieval.fusion_method is FusionMethod.RRF
    assert s.adaptive.relevance_threshold == 0.65
    assert s.adaptive.evidence_threshold == 0.70
    assert s.adaptive.max_corrective_iterations == 2


def test_flat_env_vars_populate_grouped_settings(monkeypatch):
    monkeypatch.setenv("TOP_K", "25")
    monkeypatch.setenv("RAG_STRATEGY", "corrective")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.4")
    monkeypatch.setenv("ENABLE_RERANKER", "false")
    s = Settings()
    assert s.retrieval.top_k == 25
    assert s.retrieval.rag_strategy is RagStrategy.CORRECTIVE
    assert s.retrieval.enable_reranker is False
    assert s.adaptive.relevance_threshold == 0.4


@pytest.mark.parametrize("value", ["-0.1", "1.5"])
def test_thresholds_must_be_probabilities(monkeypatch, value):
    monkeypatch.setenv("EVIDENCE_THRESHOLD", value)
    with pytest.raises(ValidationError):
        AdaptiveSettings()


def test_iteration_caps_are_bounded_to_prevent_runaway_loops(monkeypatch):
    monkeypatch.setenv("MAX_SELF_RAG_ITERATIONS", "50")
    with pytest.raises(ValidationError):
        AdaptiveSettings()


def test_rerank_top_k_cannot_exceed_candidates():
    with pytest.raises(ValidationError, match="RERANK_TOP_K"):
        RetrievalSettings(rerank_candidates=5, rerank_top_k=10)


def test_rerank_candidates_cannot_exceed_retrieval_candidates():
    with pytest.raises(ValidationError, match="RERANK_CANDIDATES"):
        RetrievalSettings(retrieval_candidates=20, rerank_candidates=30)


def test_fusion_weights_cannot_both_be_zero():
    with pytest.raises(ValidationError, match="DENSE_WEIGHT"):
        RetrievalSettings(dense_weight=0, bm25_weight=0)


def test_chunk_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError, match="OVERLAP"):
        ChunkingSettings(child_chunk_size=100, child_chunk_overlap=100)


def test_parent_must_be_at_least_child_size():
    with pytest.raises(ValidationError, match="PARENT_CHUNK_SIZE"):
        ChunkingSettings(child_chunk_size=500, parent_chunk_size=200)


def test_database_url_requires_psycopg3_driver():
    with pytest.raises(ValidationError, match="psycopg"):
        DatabaseSettings(database_url="postgresql://u:p@h/db")


def test_langfuse_enabled_without_keys_is_rejected():
    with pytest.raises(ValidationError, match="LANGFUSE"):
        ObservabilitySettings(enable_langfuse=True)


def test_api_key_checked_lazily_and_empty_string_counts_as_missing():
    llm = LLMSettings(llm_provider=LLMProvider.ANTHROPIC, anthropic_api_key="  ")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        llm.require_api_key()
    assert LLMSettings(llm_provider=LLMProvider.FAKE).require_api_key() is None


def test_openai_compatible_accepts_base_url_without_key():
    llm = LLMSettings(
        llm_provider=LLMProvider.OPENAI_COMPATIBLE, openai_base_url="http://localhost:11434/v1"
    )
    assert llm.require_api_key() is None


def test_snapshot_excludes_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:hunter2@h/db")
    snap = Settings().snapshot()
    flat = repr(snap)
    assert "sk-secret-value" not in flat
    assert "hunter2" not in flat
    assert snap["retrieval"]["top_k"] == 10
