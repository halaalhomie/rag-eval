import pytest

from app.retrieval.base import RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker, RerankingRetriever, sigmoid


def res(chunk, rank, score=1.0, text=None, components=None):
    return RetrievalResult(
        document_id=chunk.split(":")[0],
        chunk_id=chunk,
        text=text or f"text about {chunk}",
        score=score,
        rank=rank,
        retriever="hybrid",
        start_char=0,
        end_char=1,
        section="Sec",
        metadata={"title": "Title"},
        components=components,
    )


def keyword_scorer(word):
    """Fake cross-encoder: passages containing `word` score high."""
    calls = []

    def score(pairs):
        calls.append(pairs)
        return [5.0 if word in passage else -2.0 - i * 0.1 for i, (_, passage) in enumerate(pairs)]

    score.calls = calls
    return score


def test_rerank_reorders_and_records_first_stage_and_reranker_scores():
    scorer = keyword_scorer("liveness")
    reranker = CrossEncoderReranker("fake", score_fn=scorer)
    results = [
        res("a:c0", 1, 0.9, "readiness gates", components={"bm25": {"rank": 1, "score": 9.0}}),
        res("b:c0", 2, 0.8, "liveness probes restart containers"),
        res("c:c0", 3, 0.7, "volume mounts"),
    ]
    out = reranker.rerank("what does a liveness probe do", results, 2)
    assert [r.chunk_id for r in out] == ["b:c0", "a:c0"]
    assert [r.rank for r in out] == [1, 2]
    top = out[0]
    assert top.score == 5.0
    assert top.components["first_stage"] == {"rank": 2, "score": 0.8}
    assert top.components["reranker"]["probability"] == pytest.approx(sigmoid(5.0), abs=1e-6)
    # Components from the first stage (e.g. hybrid's per-retriever ranks) are preserved.
    assert out[1].components["bm25"] == {"rank": 1, "score": 9.0}


def test_passages_include_title_and_section_when_enabled():
    scorer = keyword_scorer("x")
    CrossEncoderReranker("fake", score_fn=scorer).rerank("q", [res("a:c0", 1)], 1)
    assert scorer.calls[0][0] == ("q", "Title > Sec\n\ntext about a:c0")
    scorer2 = keyword_scorer("x")
    CrossEncoderReranker("fake", score_fn=scorer2, include_context=False).rerank(
        "q", [res("a:c0", 1)], 1
    )
    assert scorer2.calls[0][0] == ("q", "text about a:c0")


def test_ties_keep_first_stage_order():
    reranker = CrossEncoderReranker("fake", score_fn=lambda pairs: [1.0] * len(pairs))
    out = reranker.rerank("q", [res("b:c0", 1), res("a:c0", 2)], 2)
    assert [r.chunk_id for r in out] == ["b:c0", "a:c0"]


def test_empty_inputs():
    reranker = CrossEncoderReranker("fake", score_fn=keyword_scorer("x"))
    assert reranker.rerank("q", [], 5) == [] and reranker.rerank("q", [res("a:c0", 1)], 0) == []


def test_sigmoid_is_numerically_stable():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(-1000.0) == pytest.approx(0.0) and sigmoid(1000.0) == pytest.approx(1.0)


class FakeBase:
    name = "hybrid"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return self.results[:k]


def test_reranking_retriever_fetches_candidates_and_returns_top_k():
    base = FakeBase(
        [res(f"d{i}:c0", i + 1, text="liveness" if i == 7 else "other") for i in range(20)]
    )
    retriever = RerankingRetriever(
        base, CrossEncoderReranker("fake", score_fn=keyword_scorer("liveness")), candidates=10
    )
    out = retriever.retrieve("liveness probe", 3, filters={"doc_category": "tasks"})
    assert base.calls == [("liveness probe", 10, {"doc_category": "tasks"})]
    assert out[0].chunk_id == "d7:c0" and out[0].components["first_stage"]["rank"] == 8
    assert len(out) == 3 and all(r.retriever == "hybrid_rerank" for r in out)
    assert retriever.retrieve("  ", 3) == []
