"""Structured output, citations, abstention and usage accounting."""

import pytest
from pydantic import BaseModel, Field

from app.config.settings import LLMSettings
from app.generation.citations import cited_numbers, format_sources, resolve_citations
from app.generation.fake import FakeLLM
from app.generation.llm import ChatMessage, LLMResponse, StructuredOutputError
from app.generation.structured import complete_structured, extract_json_object
from app.generation.usage import UsageTracker
from app.rag.prompts.generation import ABSTAIN, is_abstention
from app.retrieval.base import RetrievalResult


class Grade(BaseModel):
    relevance: float = Field(ge=0, le=1)
    contains_answer: bool


def ctx(i: int, **kw) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"d{i}",
        chunk_id=f"d{i}:c0",
        text=f"text {i}",
        score=1.0,
        rank=i,
        retriever="test",
        start_char=0,
        end_char=6,
        section=kw.get("section"),
        page=kw.get("page"),
        metadata={"title": f"Doc {i}", "url": f"https://x/{i}"},
    )


@pytest.mark.parametrize(
    "reply",
    [
        '{"relevance": 0.8, "contains_answer": true}',
        '```json\n{"relevance": 0.8, "contains_answer": true}\n```',
        'Sure! Here you go: {"relevance": 0.8, "contains_answer": true} Hope that helps.',
        '{"note": "braces } in \\"strings\\" {", "relevance": 0.8, "contains_answer": true}',
    ],
)
def test_extract_json_object_tolerates_fences_prose_and_braces_in_strings(reply):
    obj = extract_json_object(reply)
    assert obj["relevance"] == 0.8 and obj["contains_answer"] is True


def test_extract_json_object_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_structured_output_repairs_once_then_succeeds():
    llm = FakeLLM(
        replies=[
            '{"relevance": 1.7, "contains_answer": true}',
            '{"relevance": 0.7, "contains_answer": true}',
        ]
    )
    grade, responses = complete_structured(llm, [ChatMessage(role="user", content="q")], Grade)
    assert grade.relevance == 0.7 and len(responses) == 2
    repair_prompt = llm.calls[1][-1].content
    assert "invalid" in repair_prompt  # the validation error is fed back to the model
    assert "schema" in llm.calls[0][0].content  # schema described in the system prompt


def test_structured_output_raises_after_repairs_exhausted():
    llm = FakeLLM(replies=["nope", "still nope"])
    with pytest.raises(StructuredOutputError, match="Grade"):
        complete_structured(llm, [ChatMessage(role="user", content="q")], Grade)


def test_cited_numbers_order_and_dedupe():
    assert cited_numbers("A [2]. B [1, 3]. C [2][4].") == [2, 1, 3, 4]


def test_resolve_citations_maps_to_chunks_and_flags_invalid():
    contexts = [ctx(1, section="Intro"), ctx(2, page=4)]
    citations, invalid = resolve_citations("X [2]. Y [1]. Z [7].", contexts)
    assert [c.chunk_id for c in citations] == ["d2:c0", "d1:c0"]
    assert invalid == [7]
    assert citations[0].label() == "[2] Doc 2, page 4"
    assert citations[1].url == "https://x/1"


def test_format_sources_numbers_contexts_with_headers():
    out = format_sources([ctx(1, section="Intro"), ctx(2)])
    assert out.startswith("[1] Doc 1 > Intro\ntext 1")
    assert "[2] Doc 2\ntext 2" in out


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (ABSTAIN, True),
        ("I don\u2019t know based on the provided  sources", True),
        ("The grace period is 30 seconds [1].", False),
    ],
)
def test_abstention_detection(answer, expected):
    assert is_abstention(answer) is expected


def response(model, tin=1000, tout=500):
    return LLMResponse(text="", model=model, input_tokens=tin, output_tokens=tout, latency_s=0.5)


def test_usage_costs_priced_local_and_unpriced_models():
    tracker = UsageTracker(
        LLMSettings(llm_pricing={"paid": (1.0, 2.0)}, llm_local_models={"local"})
    )
    tracker.record("generate", response("paid"))
    tracker.record("grade", response("local"))
    tracker.record("judge", response("mystery"))
    s = tracker.summary()
    assert s.llm_calls == 3 and s.input_tokens == 3000 and s.output_tokens == 1500
    assert s.estimated_cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    assert s.unpriced_models == ["mystery"]  # never silently counted as free
