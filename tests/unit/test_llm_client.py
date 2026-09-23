"""OpenAI-compatible client: wire format, retry policy and error mapping (no network)."""

import httpx
import pytest

from app.config.settings import LLMSettings
from app.generation.llm import (
    ChatMessage,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServerError,
    LLMTimeoutError,
)
from app.generation.openai_compatible import OpenAICompatibleClient

MSG = [ChatMessage(role="user", content="hi")]


def ok_body(text="hello", model="m"):
    return {
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


def client(handler, retries=3, key=None):
    sleeps: list[float] = []
    settings = LLMSettings(
        openai_base_url="http://llm.test/v1", openai_api_key=key, llm_max_retries=retries
    )
    c = OpenAICompatibleClient(
        settings, transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    return c, sleeps


def test_success_parses_text_usage_and_sends_openai_payload():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=ok_body())

    c, sleeps = client(handler, key="sk-test")
    r = c.complete(MSG, model="qwen", max_tokens=50, temperature=0.2)
    assert (r.text, r.input_tokens, r.output_tokens, r.attempts) == ("hello", 12, 3, 1)
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert b'"model":"qwen"' in seen["body"] and b'"max_tokens":50' in seen["body"]
    assert sleeps == []


def test_no_auth_header_without_key():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json=ok_body())

    client(handler)[0].complete(MSG)


def test_rate_limit_is_retried_honoring_retry_after():
    responses = iter([httpx.Response(429, headers={"retry-after": "2.5"}), None])

    def handler(request):
        r = next(responses)
        return r if r is not None else httpx.Response(200, json=ok_body())

    c, sleeps = client(handler)
    assert c.complete(MSG).attempts == 2
    assert sleeps == [2.5]


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (lambda: httpx.Response(503, text="overloaded"), LLMServerError),
        (lambda: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), LLMTimeoutError),
        (lambda: (_ for _ in ()).throw(httpx.ConnectError("refused")), LLMConnectionError),
        (lambda: httpx.Response(429), LLMRateLimitError),
    ],
)
def test_retryable_failures_stop_after_max_retries(failure, error):
    calls = []

    def handler(request):
        calls.append(1)
        return failure()

    c, sleeps = client(handler, retries=2)
    with pytest.raises(error):
        c.complete(MSG)
    assert len(calls) == 3  # 1 attempt + 2 retries, then give up
    assert len(sleeps) == 2
    assert all(0 <= s <= 30 for s in sleeps)


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad model name")

    c, sleeps = client(handler)
    with pytest.raises(LLMRequestError, match="400"):
        c.complete(MSG)
    assert len(calls) == 1 and sleeps == []


def test_malformed_body_is_a_server_error():
    c, _ = client(lambda r: httpx.Response(200, json={"unexpected": True}), retries=0)
    with pytest.raises(LLMServerError, match="malformed"):
        c.complete(MSG)
