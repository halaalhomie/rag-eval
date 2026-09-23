"""Baseline RAG: embed query -> vector search -> top-k chunks -> LLM -> cited answer.

Deliberately minimal: no rewriting, no reranking, no grading, one LLM call. Every other
strategy is measured against it, so it must stay free of extras.
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from app.generation.citations import resolve_citations
from app.generation.llm import LLMClient
from app.generation.usage import UsageTracker
from app.rag.prompts.generation import ABSTAIN, answer_messages, is_abstention
from app.rag.types import RAGResult, Timings
from app.retrieval.base import MetadataFilter, Retriever

logger = logging.getLogger(__name__)


class BaselineRAG:
    name = "baseline"

    def __init__(self, retriever: Retriever, llm: LLMClient, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def run(
        self, query: str, *, top_k: int | None = None, filters: MetadataFilter | None = None
    ) -> RAGResult:
        started = time.perf_counter()
        usage = UsageTracker(self.settings.llm)
        k = top_k or self.settings.retrieval.top_k

        contexts = self.retriever.retrieve(query, k, filters)
        retrieval_s = time.perf_counter() - started

        if not contexts:
            # Nothing to ground an answer in: abstain without spending an LLM call.
            logger.info("Empty retrieval; abstaining", extra={"query": query})
            answer, generation_s = ABSTAIN, 0.0
        else:
            gen_started = time.perf_counter()
            response = self.llm.complete(answer_messages(query, contexts))
            usage.record("generate", response)
            answer = response.text.strip()
            generation_s = time.perf_counter() - gen_started

        citations, invalid = resolve_citations(answer, contexts)
        return RAGResult(
            query=query,
            strategy=self.name,
            answer=answer,
            abstained=is_abstention(answer),
            citations=citations,
            invalid_citations=invalid,
            contexts=contexts,
            timings=Timings(
                retrieval_s=round(retrieval_s, 4),
                generation_s=round(generation_s, 4),
                total_s=round(time.perf_counter() - started, 4),
            ),
            usage=usage.summary(),
            llm_calls=usage.records,
        )
