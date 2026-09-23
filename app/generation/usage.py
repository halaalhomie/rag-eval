"""Per-request accounting of LLM calls: tokens, latency and estimated cost.

Every LLM call a pipeline makes is recorded with a `purpose` (generate, grade, rewrite,
judge, ...). The quality-vs-cost comparison between strategies is built from these records.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.settings import LLMSettings
from app.generation.llm import LLMResponse


class LLMCallRecord(BaseModel):
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int
    cost_usd: float | None  # None = model has no configured price


class UsageSummary(BaseModel):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    unpriced_models: list[str] = Field(default_factory=list)


class UsageTracker:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.records: list[LLMCallRecord] = []

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        if model in self.settings.llm_pricing:
            price_in, price_out = self.settings.llm_pricing[model]
            return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        if model in self.settings.llm_local_models:
            return 0.0
        return None

    def record(self, purpose: str, response: LLMResponse) -> LLMCallRecord:
        rec = LLMCallRecord(
            purpose=purpose,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_s=response.latency_s,
            attempts=response.attempts,
            cost_usd=self.cost(response.model, response.input_tokens, response.output_tokens),
        )
        self.records.append(rec)
        return rec

    def summary(self) -> UsageSummary:
        unpriced = sorted({r.model for r in self.records if r.cost_usd is None})
        return UsageSummary(
            llm_calls=len(self.records),
            input_tokens=sum(r.input_tokens for r in self.records),
            output_tokens=sum(r.output_tokens for r in self.records),
            llm_latency_s=round(sum(r.latency_s for r in self.records), 4),
            estimated_cost_usd=round(sum(r.cost_usd or 0.0 for r in self.records), 6),
            unpriced_models=unpriced,
        )
