"""Cross-encoder reranking.

A bi-encoder (dense retrieval) embeds query and passage separately; a cross-encoder reads
them *together*, so it can model their interaction directly. That is much more accurate
and much more expensive, so it is applied only to a first-stage candidate list:

    first stage (e.g. hybrid) -> top RERANK_CANDIDATES -> cross-encoder -> top k

Every reranked result records (in `components`):
- first_stage: its rank and score before reranking (and the first stage's own components)
- reranker:    the raw cross-encoder score and its sigmoid, a 0-1 relevance estimate
               (used later by corrective retrieval as a document-relevance signal)
and its `rank` is the final rank.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache

from app.config import Settings
from app.ingestion.pipeline import embedding_input
from app.retrieval.base import MetadataFilter, RetrievalResult, Retriever

logger = logging.getLogger(__name__)

# (query, passage) pairs -> raw relevance scores. Injectable for tests.
ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


class RerankerError(RuntimeError):
    pass


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        include_context: bool = True,
        score_fn: ScoreFn | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.include_context = include_context
        self._score_fn = score_fn
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        from sentence_transformers import CrossEncoder

        try:
            model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        except Exception as exc:
            raise RerankerError(f"cannot load reranker {self.model_name}: {exc}") from exc
        logger.info("Loaded reranker %s on %s", self.model_name, self.device)
        return model

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        if self._score_fn is not None:
            return list(self._score_fn(pairs))
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        import torch

        try:
            # Raw logits. Note: activation_fn=None does NOT mean "no activation" here; it
            # falls back to the model default (Sigmoid), so pass Identity explicitly. The
            # sigmoid is applied separately so both raw score and probability are recorded.
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RerankerError(f"reranking failed: {exc}") from exc
        return [float(x) for x in scores]

    def passage(self, r: RetrievalResult) -> str:
        return embedding_input(r.title, r.section, r.text) if self.include_context else r.text

    def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        if not results or top_k <= 0:
            return []
        scores = self.score([(query, self.passage(r)) for r in results])
        order = sorted(range(len(results)), key=lambda i: (-scores[i], results[i].rank))
        out = []
        for final_rank, i in enumerate(order[:top_k], start=1):
            r = results[i]
            components = dict(r.components or {})
            components["first_stage"] = {"rank": r.rank, "score": round(r.score, 6)}
            components["reranker"] = {
                "score": round(scores[i], 6),
                "probability": round(sigmoid(scores[i]), 6),
            }
            out.append(
                r.model_copy(
                    update={"score": scores[i], "rank": final_rank, "components": components}
                )
            )
        return out


class RerankingRetriever:
    """Wraps any retriever: fetch `candidates` results, rerank, return the top k."""

    def __init__(
        self,
        base: Retriever,
        reranker: CrossEncoderReranker,
        candidates: int = 30,
        name: str | None = None,
    ):
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.name = name or f"{base.name}_rerank"

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> list[RetrievalResult]:
        if k <= 0 or not query.strip():
            return []
        candidates = self.base.retrieve(query, max(k, self.candidates), filters)
        reranked = self.reranker.rerank(query, candidates, k)
        return [r.model_copy(update={"retriever": self.name}) for r in reranked]


@lru_cache(maxsize=2)
def _shared_reranker(
    model: str, device: str, batch: int, max_length: int, ctx: bool
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model, device=device, batch_size=batch, max_length=max_length, include_context=ctx
    )


def get_reranker(settings: Settings) -> CrossEncoderReranker:
    """Process-wide instance per configuration (weights load once, on first use)."""
    r = settings.retrieval
    return _shared_reranker(
        r.reranker_model,
        r.reranker_device,
        r.reranker_batch_size,
        r.reranker_max_length,
        settings.embedding.embedding_include_context,
    )
