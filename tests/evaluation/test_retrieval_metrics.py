"""Metric definitions checked against hand-computed values."""

import math

import pytest

from app.evaluation.metrics.retrieval import (
    evidence_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKING = ["a", "x", "b", "y", "c"]
RELEVANT = {"a", "b", "c", "d"}


def test_recall_and_precision():
    assert recall_at_k(RANKING, RELEVANT, 1) == 0.25
    assert recall_at_k(RANKING, RELEVANT, 3) == 0.5
    assert recall_at_k(RANKING, RELEVANT, 5) == 0.75
    assert precision_at_k(RANKING, RELEVANT, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKING, RELEVANT, 10) == 0.3  # denominator is K, not len(ranking)


def test_hit_and_reciprocal_rank():
    assert hit_at_k(["x", "a"], {"a"}, 1) == 0.0
    assert hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_ndcg_matches_hand_computation():
    # relevant at ranks 1, 3, 5; ideal: 3 relevant items at ranks 1, 2, 3 (K=5, |rel|=4)
    dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    assert ndcg_at_k(RANKING, RELEVANT, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_ignores_duplicate_hits():
    assert ndcg_at_k(["a", "a"], {"a", "b"}, 2) == ndcg_at_k(["a", "z"], {"a", "b"}, 2)


def test_evidence_recall_counts_spans_not_chunks():
    # span 1 covered by two overlapping chunks, span 2 by one chunk
    spans = [{"c1", "c2"}, {"c9"}]
    assert evidence_recall_at_k(["c2", "x"], spans, 2) == 0.5  # one span found
    assert evidence_recall_at_k(["c1", "c9"], spans, 2) == 1.0
    # whereas chunk recall would penalize finding only one of the two duplicate chunks
    assert recall_at_k(["c1", "c9"], {"c1", "c2", "c9"}, 2) == pytest.approx(2 / 3)


def test_unmapped_span_counts_as_miss():
    assert evidence_recall_at_k(["c1"], [{"c1"}, set()], 5) == 0.5


@pytest.mark.parametrize(
    "fn",
    [
        lambda: recall_at_k(["a"], set(), 1),
        lambda: ndcg_at_k(["a"], set(), 1),
        lambda: evidence_recall_at_k(["a"], [], 1),
        lambda: precision_at_k(["a"], {"a"}, 0),
    ],
)
def test_undefined_cases_raise(fn):
    with pytest.raises(ValueError):
        fn()


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    from app.evaluation.retrieval.runner import bootstrap_ci

    values = [1.0] * 70 + [0.0] * 30
    low, high = bootstrap_ci(values)
    assert low < 0.7 < high and high - low < 0.25
    assert bootstrap_ci(values) == [low, high]
    assert bootstrap_ci([0.5] * 10) == [0.5, 0.5]
