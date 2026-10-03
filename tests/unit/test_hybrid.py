import pytest

from app.config import FusionMethod
from app.retrieval.base import RetrievalResult
from app.retrieval.hybrid import HybridRetriever, linear_fuse, rrf_fuse


def res(chunk, rank, score, retriever):
    return RetrievalResult(
        document_id=chunk.split(":")[0],
        chunk_id=chunk,
        text=chunk,
        score=score,
        rank=rank,
        retriever=retriever,
        start_char=0,
        end_char=1,
    )


def ranked(retriever, chunks, scores=None):
    scores = scores or [1.0 / (i + 1) for i in range(len(chunks))]
    return [
        res(c, i + 1, s, retriever) for i, (c, s) in enumerate(zip(chunks, scores, strict=True))
    ]


class ListRetriever:
    def __init__(self, name, chunks, scores=None):
        self.name = name
        self.chunks, self.scores = chunks, scores
        self.calls = []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return ranked(self.name, self.chunks, self.scores)[:k]


# ---------------------------------------------------------------- RRF
def test_rrf_matches_hand_computation():
    lists = {"dense": ranked("dense", ["a", "b", "c"]), "bm25": ranked("bm25", ["c", "a"])}
    fused = dict(rrf_fuse(lists, {"dense": 1.0, "bm25": 1.0}, k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    order = [c for c, _ in rrf_fuse(lists, {"dense": 1.0, "bm25": 1.0}, k=60)]
    assert order == ["a", "c", "b"]  # a: ranks (1,2) beats c: ranks (3,1) by a hair


def test_rrf_weights_shift_the_ranking():
    lists = {"dense": ranked("dense", ["a", "b"]), "bm25": ranked("bm25", ["b", "a"])}
    assert rrf_fuse(lists, {"dense": 0.7, "bm25": 0.3})[0][0] == "a"
    assert rrf_fuse(lists, {"dense": 0.3, "bm25": 0.7})[0][0] == "b"


def test_rrf_ties_break_by_chunk_id():
    lists = {"dense": ranked("dense", ["b"]), "bm25": ranked("bm25", ["a"])}
    assert [c for c, _ in rrf_fuse(lists, {"dense": 1, "bm25": 1})] == ["a", "b"]


# ---------------------------------------------------------------- linear
def test_linear_fusion_min_max_normalizes_each_list():
    lists = {
        "dense": ranked("dense", ["a", "b", "c"], [0.9, 0.8, 0.7]),
        "bm25": ranked("bm25", ["c", "a"], [20.0, 10.0]),
    }
    fused = dict(linear_fuse(lists, {"dense": 0.5, "bm25": 0.5}))
    assert fused["a"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)
    assert fused["b"] == pytest.approx(0.5 * 0.5)
    assert fused["c"] == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)


def test_linear_fusion_single_result_list_gets_full_score():
    lists = {"dense": ranked("dense", ["a"], [0.3]), "bm25": []}
    assert linear_fuse(lists, {"dense": 1.0, "bm25": 1.0}) == [("a", 1.0)]


# ---------------------------------------------------------------- retriever
def test_hybrid_fetches_candidates_fuses_and_records_components():
    dense = ListRetriever("dense", ["a", "b", "c"], [0.9, 0.8, 0.7])
    bm25 = ListRetriever("bm25", ["c", "d"], [12.0, 3.0])
    hybrid = HybridRetriever(
        {"dense": dense, "bm25": bm25}, {"dense": 1.0, "bm25": 1.0}, candidates=50
    )
    out = hybrid.retrieve("q", 3, filters={"doc_category": "tasks"})
    assert dense.calls == [("q", 50, {"doc_category": "tasks"})]  # deep candidates, filters pass
    # b (dense rank 2) and d (bm25 rank 2) tie at 1/62; the tie breaks by chunk id.
    assert [r.chunk_id for r in out] == ["c", "a", "b"]
    assert [r.rank for r in out] == [1, 2, 3]
    assert all(r.retriever == "hybrid" for r in out)
    assert out[0].components == {
        "dense": {"rank": 3, "score": 0.7},
        "bm25": {"rank": 1, "score": 12.0},
    }
    assert out[1].components == {"dense": {"rank": 1, "score": 0.9}}
    assert out[0].score == pytest.approx(1 / 63 + 1 / 61)


def test_hybrid_survives_one_empty_retriever_and_dedupes():
    hybrid = HybridRetriever(
        {"dense": ListRetriever("dense", ["a", "b"]), "bm25": ListRetriever("bm25", [])},
        {"dense": 0.5, "bm25": 0.5},
    )
    assert [r.chunk_id for r in hybrid.retrieve("q", 5)] == ["a", "b"]


def test_hybrid_linear_method_and_validation():
    hybrid = HybridRetriever(
        {
            "dense": ListRetriever("dense", ["a", "b"], [0.9, 0.1]),
            "bm25": ListRetriever("bm25", ["b", "a"], [5.0, 4.0]),
        },
        {"dense": 0.8, "bm25": 0.2},
        method=FusionMethod.LINEAR,
    )
    assert hybrid.retrieve("q", 1)[0].chunk_id == "a"
    assert hybrid.retrieve("  ", 5) == [] and hybrid.retrieve("q", 0) == []
    with pytest.raises(ValueError):
        HybridRetriever({"dense": ListRetriever("dense", [])}, {"dense": 0.0})
