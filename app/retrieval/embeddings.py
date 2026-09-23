"""Embedding models.

`SentenceTransformerEmbedder` wraps any sentence-transformers model from Hugging Face.
Embeddings are L2-normalized, so cosine distance and inner product rank identically.
Query and passage encodings are asymmetric for instruction-tuned models such as BGE:
queries get EMBEDDING_QUERY_PREFIX and passages do not.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.model_name = settings.embedding_model
        self.dim = settings.embedding_dim
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        """Loaded on first use: importing torch and loading weights takes seconds and
        about 0.5 GB, which tests and non-retrieval code paths should not pay for."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self):
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(self.model_name, device=self.settings.embedding_device)
        except Exception as exc:  # network, missing weights, bad device
            raise EmbeddingError(f"cannot load embedding model {self.model_name}: {exc}") from exc
        actual = model.get_sentence_embedding_dimension()
        if actual != self.dim:
            raise EmbeddingError(
                f"{self.model_name} produces {actual}-d vectors but EMBEDDING_DIM={self.dim}; "
                "fix EMBEDDING_DIM and re-initialize the database"
            )
        logger.info("Loaded embedding model %s (%d-d)", self.model_name, actual)
        return model

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def token_count(self, text: str) -> int:
        """Exact token count under this model's tokenizer (used to audit truncation)."""
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self.model.encode(
                list(texts),
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"embedding failed: {exc}") from exc
        return vectors.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.settings.embedding_query_prefix + text])[0]


@lru_cache(maxsize=4)
def _cached(
    model: str, dim: int, device: str, prefix: str, batch: int
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        EmbeddingSettings(
            embedding_model=model,
            embedding_dim=dim,
            embedding_device=device,
            embedding_query_prefix=prefix,
            embedding_batch_size=batch,
        )
    )


def get_embedder(settings: EmbeddingSettings) -> SentenceTransformerEmbedder:
    """Process-wide instance per model configuration (weights are loaded once)."""
    return _cached(
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_device,
        settings.embedding_query_prefix,
        settings.embedding_batch_size,
    )
