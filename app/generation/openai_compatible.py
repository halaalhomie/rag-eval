"""Client for any OpenAI-compatible chat-completions server.

Covers local servers (mlx_lm.server, Ollama, vLLM, llama.cpp) and hosted APIs (OpenAI,
Groq, Gemini, Hugging Face Inference Providers). Talks HTTP directly with httpx: the wire
format is small, and owning it keeps retry and error mapping explicit.

Retries: timeouts, connection errors, 429 and 5xx are retried with exponential backoff and
full jitter, honoring `Retry-After`. Other 4xx responses fail immediately. The attempt
count is bounded by LLM_MAX_RETRIES.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import httpx

from app.config.settings import LLMSettings
from app.generation.llm import (
    ChatMessage,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponse,
    LLMServerError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 30.0


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: LLMSettings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = settings.require_api_key()
        headers = {"Authorization": f"Bearer {key.get_secret_value()}"} if key else {}
        self.settings = settings
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=(settings.openai_base_url or "https://api.openai.com/v1").rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=10.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        s = self.settings
        payload = {
            "model": model or s.llm_model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens or s.llm_max_tokens,
            "temperature": s.llm_temperature if temperature is None else temperature,
        }
        attempts = s.llm_max_retries + 1
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                response = self._send(payload)
                response.attempts = attempt
                response.latency_s = time.perf_counter() - started
                return response
            except LLMError as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                delay = self._backoff(attempt, exc)
                logger.warning(
                    "LLM call failed (%s), retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt,
                    attempts - 1,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(attempt: int, exc: LLMError) -> float:
        if isinstance(exc, LLMRateLimitError) and exc.retry_after_s is not None:
            return min(exc.retry_after_s, _BACKOFF_CAP_S)
        return random.uniform(0, min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2 ** (attempt - 1)))

    def _send(self, payload: dict) -> LLMResponse:
        try:
            r = self._http.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"timed out after {self.settings.llm_timeout_s}s") from exc
        except httpx.TransportError as exc:
            raise LLMConnectionError(f"cannot reach {self._http.base_url}: {exc}") from exc

        if r.status_code == 429:
            raise LLMRateLimitError("rate limited", _retry_after(r.headers.get("retry-after")))
        if r.status_code >= 500:
            raise LLMServerError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise LLMRequestError(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            body = r.json()
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            return LLMResponse(
                text=choice["message"]["content"] or "",
                model=body.get("model") or payload["model"],
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=0.0,
                finish_reason=choice.get("finish_reason"),
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMServerError(f"malformed response: {r.text[:200]}") from exc


def _retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header; HTTP-date values are ignored (use backoff)."""
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:
        return None
