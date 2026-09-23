import pytest

from app.config import Settings
from app.generation.fake import FakeLLM
from app.generation.llm import LLMServerError
from app.rag.prompts.generation import ABSTAIN, REMINDER
from app.rag.strategies.baseline import BaselineRAG
from app.retrieval.base import RetrievalResult


class StaticRetriever:
    name = "static"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return self.results[:k]


def result(i):
    return RetrievalResult(
        document_id=f"d{i}",
        chunk_id=f"d{i}:c{i}",
        text=f"Fact number {i}.",
        score=0.9,
        rank=i,
        retriever="static",
        start_char=0,
        end_char=15,
        metadata={"title": f"Doc {i}", "url": f"https://docs/{i}"},
    )


def test_baseline_generates_cited_answer_from_top_k():
    retriever = StaticRetriever([result(1), result(2), result(3)])
    llm = FakeLLM(replies=["Fact two holds [2]. Also fact one [1]."])
    out = BaselineRAG(retriever, llm, Settings()).run("q?", top_k=2, filters={"a": 1})

    assert retriever.calls == [("q?", 2, {"a": 1})]
    assert [c.chunk_id for c in out.citations] == ["d2:c2", "d1:c1"]
    assert not out.abstained and out.invalid_citations == []
    assert out.usage.llm_calls == 1 and out.llm_calls[0].purpose == "generate"
    assert len(out.contexts) == 2 and out.strategy == "baseline"
    prompt = llm.calls[0][-1].content
    assert "[1] Doc 1\nFact number 1." in prompt and "[3]" not in prompt
    assert prompt.rstrip().endswith(REMINDER)  # citation rule restated after sources


def test_baseline_uses_configured_top_k_by_default():
    retriever = StaticRetriever([result(i) for i in range(1, 20)])
    BaselineRAG(retriever, FakeLLM(), Settings(retrieval={"top_k": 7})).run("q")
    assert retriever.calls[0][1] == 7


def test_empty_retrieval_abstains_without_calling_llm():
    llm = FakeLLM()
    out = BaselineRAG(StaticRetriever([]), llm, Settings()).run("anything")
    assert out.answer == ABSTAIN and out.abstained
    assert llm.calls == [] and out.usage.llm_calls == 0


def test_llm_failure_propagates():
    llm = FakeLLM(replies=[LLMServerError("down")])
    with pytest.raises(LLMServerError):
        BaselineRAG(StaticRetriever([result(1)]), llm, Settings()).run("q")
