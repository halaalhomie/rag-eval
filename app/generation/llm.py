"""Provider-neutral LLM interface.

Pipelines depend only on `LLMClient.complete()`. Providers translate to their wire format
and map transport failures onto the error hierarchy below, so retry policy and pipeline
error handling never contain provider-specific code.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str | None = None
    attempts: int = 1


class LLMError(RuntimeError):
    """Base class. `retryable` drives the client's retry loop."""

    retryable = False


class LLMTimeoutError(LLMError):
    retryable = True


class LLMConnectionError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):
    retryable = True

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMServerError(LLMError):
    retryable = True


class LLMRequestError(LLMError):
    """4xx other than 429: the request itself is wrong, retrying cannot help."""


class StructuredOutputError(LLMError):
    """The model did not produce output matching the requested schema."""


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...
