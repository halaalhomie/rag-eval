import math

import pytest

from app.retrieval.bm25 import BM25Index, _Doc
from app.retrieval.text import Analyzer, stem


def doc(i, text, **meta):
    return _Doc(f"d{i}:c0", f"d{i}", None, text, None, None, 0, len(text), meta)


# ---------------------------------------------------------------- analyzer
def test_words_are_lowercased_stopworded_and_stemmed():
    # Stems are matching keys, not words: "probes" -> "prob" on both query and index side.
    assert Analyzer(compounds=False, camel_parts=False)("The Probes are restarting") == [
        "prob",
        "restart",
    ]
    assert Analyzer(compounds=False, camel_parts=False, stemming=False)("Probes") == ["probes"]


def test_compound_identifiers_are_kept_whole_alongside_parts():
    tokens = Analyzer(camel_parts=False, stemming=False)("run kube-apiserver --grace-period=0")
    assert {"kube", "apiserver", "kube-apiserver", "grace-period"} <= set(tokens)
    assert "spec.replicas" in Analyzer()("set .spec.replicas to 3")


def test_camelcase_identifiers_add_their_components():
    tokens = Analyzer(compounds=False, stemming=False)("PodDisruptionBudget limits evictions")
    assert {"poddisruptionbudget", "pod", "disruption", "budget"} <= set(tokens)
    # Ordinary capitalized words are not split into "parts".
    assert Analyzer(compounds=False, stemming=False)("Pods") == ["pods"]


def test_analyzer_name_reflects_options():
    assert Analyzer().name() == "words+compound+camel+stem"
    assert Analyzer(False, False, False).name() == "words"


def test_stem_matches_validator_rules():
    assert {stem(w) for w in ("require", "requires", "required", "requiring")} == {"requir"}


# ---------------------------------------------------------------- scoring
def test_bm25_score_matches_hand_computation():
    plain = Analyzer(compounds=False, camel_parts=False, stemming=False)
    docs = [doc(0, "liveness probe probe"), doc(1, "readiness probe"), doc(2, "volume mount")]
    index = BM25Index(docs, plain, include_context=False, k1=1.2, b=0.75)
    n, avgdl = 3, (3 + 2 + 2) / 3
    df = 2  # "probe" occurs in docs 0 and 1
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    tf, dl = 2, 3
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
    top_i, top_score = index.search("probe", 3)[0]
    assert top_i == 0 and top_score == pytest.approx(expected)
    assert [i for i, _ in index.search("probe", 3)] == [0, 1]  # doc 2 has no match


def test_rare_terms_outweigh_common_terms():
    plain = Analyzer(compounds=False, camel_parts=False, stemming=False)
    docs = [doc(i, "pod pod pod") for i in range(8)] + [doc(8, "pod tolerations")]
    index = BM25Index(docs, plain, include_context=False, k1=1.2, b=0.75)
    assert index.search("pod tolerations", 1)[0][0] == 8


def test_b_zero_disables_length_normalization():
    plain = Analyzer(compounds=False, camel_parts=False, stemming=False)
    docs = [doc(0, "secret"), doc(1, "secret " + "filler " * 50)]
    with_norm = BM25Index(docs, plain, False, k1=1.2, b=0.75).search("secret", 2)
    without = BM25Index(docs, plain, False, k1=1.2, b=0.0).search("secret", 2)
    assert with_norm[0][1] > with_norm[1][1]  # the short doc wins with normalization
    assert without[0][1] == pytest.approx(without[1][1])


def test_ties_are_broken_deterministically_by_chunk_id():
    plain = Analyzer(compounds=False, camel_parts=False, stemming=False)
    docs = [doc(2, "node"), doc(1, "node"), doc(3, "node")]
    index = BM25Index(docs, plain, False, 1.2, 0.75)
    assert [docs[i].chunk_id for i, _ in index.search("node", 3)] == ["d1:c0", "d2:c0", "d3:c0"]


def test_unknown_and_repeated_query_terms():
    plain = Analyzer(compounds=False, camel_parts=False, stemming=False)
    index = BM25Index([doc(0, "service")], plain, False, 1.2, 0.75)
    assert index.search("zebra", 5) == []
    assert index.search("service service", 1) == index.search("service", 1)


def test_context_prefix_is_indexed_when_enabled():
    d = _Doc(
        "d0:c0",
        "d0",
        None,
        "Set the field to true.",
        "Pod Lifecycle",
        None,
        0,
        22,
        {"title": "Probes"},
    )
    plain = Analyzer(compounds=False, camel_parts=False, stemming=False)
    assert BM25Index([d], plain, include_context=True, k1=1.2, b=0.75).search("probes", 1)
    assert BM25Index([d], plain, include_context=False, k1=1.2, b=0.75).search("probes", 1) == []
