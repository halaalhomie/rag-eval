"""Deterministic LLM for tests and offline development.

Responses come from a responder function (messages -> text), or from a queue of canned
replies. Every call is recorded so tests can assert on prompts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from app.generation.llm import ChatMessage, LLMError, LLMResponse
from app.ingestion.tokens import estimate_tokens

Responder = Callable[[list[ChatMessage]], str]


def _default_responder(messages: list[ChatMessage]) -> str:
    return "I don't know based on the provided sources."


class FakeLLM:
    def __init__(
        self,
        responder: Responder | None = None,
        replies: Iterable[str | Exception] | None = None,
        model: str = "fake-llm",
    ):
        self.responder = responder or _default_responder
        self.replies: deque[str | Exception] = deque(replies or [])
        self.model = model
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        reply = self.replies.popleft() if self.replies else self.responder(messages)
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, str):
            raise LLMError(f"fake reply must be str, got {type(reply).__name__}")
        return LLMResponse(
            text=reply,
            model=model or self.model,
            input_tokens=sum(estimate_tokens(m.content) for m in messages),
            output_tokens=estimate_tokens(reply),
            latency_s=0.0,
            finish_reason="stop",
        )
