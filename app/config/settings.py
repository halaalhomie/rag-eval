"""Centralized, validated configuration.

Every tunable (thresholds, iteration caps, fusion weights, model names) lives here and is
read from environment variables / `.env`. Nothing downstream should hardcode these values:
nodes and retrievers receive a settings object (or a slice of it) explicitly, which is also
what lets the experiment runner override configuration per experiment.

Settings are split into groups for readability, but env var names stay flat
(`TOP_K`, `RELEVANCE_THRESHOLD`, ...) to keep `.env` files simple.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from typing import Any, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagStrategy(StrEnum):
    """Named pipeline configurations. Each maps to one benchmarkable experiment."""

    BASELINE = "baseline"  # dense top-k -> LLM, nothing else
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    HYBRID_RERANK_REWRITE = "hybrid_rerank_rewrite"
    CORRECTIVE = "corrective"
    SELF_RAG = "self_rag"
    ADAPTIVE = "adaptive"  # query analyzer picks one of the above per query


class SimilarityMetric(StrEnum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class FusionMethod(StrEnum):
    RRF = "rrf"  # weighted reciprocal rank fusion
    LINEAR = "linear"  # weighted sum of min-max normalized scores


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI, vLLM, Ollama, etc.
    FAKE = "fake"  # deterministic, for tests and offline development


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


# RAG_FORGE_ENV_FILE lets tests (and alternate deployments) point away from ./.env.
_BASE_CONFIG = SettingsConfigDict(
    env_file=os.environ.get("RAG_FORGE_ENV_FILE", ".env") or None,
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class DatabaseSettings(BaseSettings):
    model_config = _BASE_CONFIG

    database_url: str = Field(
        default="postgresql+psycopg://ragforge:ragforge@localhost:5434/ragforge",
        description="SQLAlchemy URL. Must use the psycopg (v3) driver.",
    )
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_echo: bool = False

    @model_validator(mode="after")
    def _check_driver(self) -> Self:
        if not self.database_url.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must start with 'postgresql+psycopg://'")
        return self


class EmbeddingSettings(BaseSettings):
    model_config = _BASE_CONFIG

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, ge=8, le=4096)
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    # BGE models expect an instruction prefix on queries (not on passages).
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_device: str = "cpu"


class ChunkingSettings(BaseSettings):
    model_config = _BASE_CONFIG

    # Child chunks are what gets indexed; parents are what can be sent to the LLM.
    child_chunk_size: int = Field(default=400, ge=50, le=4000, description="tokens (approx)")
    child_chunk_overlap: int = Field(default=60, ge=0, le=1000)
    parent_chunk_size: int = Field(default=1600, ge=100, le=16000)

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than CHILD_CHUNK_SIZE")
        if self.parent_chunk_size < self.child_chunk_size:
            raise ValueError("PARENT_CHUNK_SIZE must be >= CHILD_CHUNK_SIZE")
        return self


class RetrievalSettings(BaseSettings):
    model_config = _BASE_CONFIG

    rag_strategy: RagStrategy = RagStrategy.ADAPTIVE
    top_k: int = Field(default=10, ge=1, le=200, description="final contexts passed on")
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE

    enable_bm25: bool = True
    enable_reranker: bool = True
    enable_parent_child: bool = True
    enable_query_rewrite: bool = True
    enable_multi_query: bool = False

    fusion_method: FusionMethod = FusionMethod.RRF
    dense_weight: float = Field(default=0.5, ge=0.0)
    bm25_weight: float = Field(default=0.5, ge=0.0)
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant (Cormack et al.)")
    # How many candidates each retriever contributes before fusion.
    retrieval_candidates: int = Field(default=50, ge=1, le=1000)

    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = Field(default=30, ge=1, le=500)
    rerank_top_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("DENSE_WEIGHT + BM25_WEIGHT must be > 0")
        if self.rerank_top_k > self.rerank_candidates:
            raise ValueError("RERANK_TOP_K must be <= RERANK_CANDIDATES")
        if self.rerank_candidates > self.retrieval_candidates:
            raise ValueError("RERANK_CANDIDATES must be <= RETRIEVAL_CANDIDATES")
        return self


class AdaptiveSettings(BaseSettings):
    """Thresholds and loop caps for corrective / Self-RAG-inspired behavior."""

    model_config = _BASE_CONFIG

    relevance_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    evidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    faithfulness_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # Hard upper bounds (le=5) so a bad config can never create a runaway loop.
    max_corrective_iterations: int = Field(default=2, ge=0, le=5)
    max_self_rag_iterations: int = Field(default=2, ge=0, le=5)
    max_regenerations: int = Field(default=1, ge=0, le=3)


class LLMSettings(BaseSettings):
    model_config = _BASE_CONFIG

    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-sonnet-5"
    # Cheaper model for high-volume structured calls (grading, classification, claims).
    llm_fast_model: str = "claude-haiku-4-5-20251001"
    # Kept separate so the judge can differ from the generator (reduces self-preference bias).
    judge_model: str = "claude-sonnet-5"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=16, le=32000)
    llm_timeout_s: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None

    def require_api_key(self) -> SecretStr | None:
        """Called when a provider client is built, not at import time, so tests and
        offline tooling can load settings without secrets."""

        def missing(key: SecretStr | None) -> bool:  # `KEY=` in .env parses as ""
            return key is None or not key.get_secret_value().strip()

        if self.llm_provider is LLMProvider.ANTHROPIC and missing(self.anthropic_api_key):
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        if (
            self.llm_provider is LLMProvider.OPENAI_COMPATIBLE
            and missing(self.openai_api_key)
            and not self.openai_base_url
        ):
            raise ValueError(
                "LLM_PROVIDER=openai_compatible requires OPENAI_API_KEY or OPENAI_BASE_URL"
            )
        if self.llm_provider is LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        return self.openai_api_key


class ObservabilitySettings(BaseSettings):
    model_config = _BASE_CONFIG

    enable_langfuse: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_json: bool = False

    @model_validator(mode="after")
    def _check_langfuse(self) -> Self:
        if self.enable_langfuse and not (self.langfuse_public_key and self.langfuse_secret_key):
            raise ValueError(
                "ENABLE_LANGFUSE=true requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
            )
        return self


class Settings(BaseSettings):
    model_config = _BASE_CONFIG

    app_name: str = "rag-forge"
    environment: Environment = Environment.DEVELOPMENT

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    def snapshot(self) -> dict[str, Any]:
        """Secret-free dump of the effective config, recorded with every experiment."""
        return self.model_dump(
            mode="json",
            exclude={
                "database": {"database_url"},
                "llm": {"anthropic_api_key", "openai_api_key"},
                "observability": {"langfuse_public_key", "langfuse_secret_key"},
            },
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
