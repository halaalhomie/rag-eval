"""Dataset schema, span labelling, validators, splits and the retrieval runner."""

import pytest

from app.evaluation.datasets import validators as V
from app.evaluation.datasets.generator import assign_splits
from app.evaluation.datasets.labels import ChunkIndex, ChunkSpan
from app.evaluation.datasets.schema import (
    EvalItem,
    EvidenceSpan,
    Provenance,
    QuestionType,
    load_dataset,
    save_dataset,
)
from app.evaluation.retrieval.runner import evaluate_retrieval
from app.ingestion.chunking import sentence_spans
from app.retrieval.base import RetrievalResult

PROV = Provenance(method="synthetic_llm", generator_model="m")


def span(doc="d1", start=100, end=150):
    return EvidenceSpan(document_id=doc, start_char=start, end_char=end, quote="x" * (end - start))


def item(i=0, qtype=QuestionType.SIMPLE_FACTUAL, evidence=None, split="test"):
    evidence = [span()] if evidence is None else evidence
    return EvalItem(
        id=f"q{i}",
        question=f"What is thing number {i}?",
        ground_truth_answer="It is.",
        question_type=qtype,
        difficulty="easy",
        split=split,
        evidence=evidence,
        relevant_document_ids=sorted({e.document_id for e in evidence}),
        provenance=PROV,
    )


# ---------------------------------------------------------------- schema
def test_answerable_items_require_evidence_and_unanswerable_forbid_it():
    with pytest.raises(ValueError, match="require evidence"):
        item(evidence=[])
    with pytest.raises(ValueError, match="must not have evidence"):
        item(qtype=QuestionType.UNANSWERABLE)
    assert not item(qtype=QuestionType.UNANSWERABLE, evidence=[]).has_evidence


def test_quote_length_must_match_span():
    with pytest.raises(ValueError, match="quote length"):
        EvidenceSpan(document_id="d", start_char=0, end_char=10, quote="short")


def test_relevant_documents_must_match_evidence():
    with pytest.raises(ValueError, match="relevant_document_ids"):
        EvalItem(**{**item().model_dump(), "relevant_document_ids": ["other"]})


def test_roundtrip_filters_and_duplicate_ids(tmp_path):
    path = tmp_path / "ds.jsonl"
    items = [item(0, split="dev"), item(1), item(2, QuestionType.UNANSWERABLE, [])]
    version = save_dataset(items, path)
    assert len(version) == 12
    assert [i.id for i in load_dataset(path, split="dev")] == ["q0"]
    assert [i.id for i in load_dataset(path, types={QuestionType.UNANSWERABLE})] == ["q2"]
    save_dataset([item(0), item(0)], path)
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


# ---------------------------------------------------------------- labels
def test_span_relevance_uses_coverage_and_survives_rechunking():
    it = item(evidence=[span("d1", 100, 200)])
    chunking_a = ChunkIndex({"d1": [ChunkSpan("a0", "d1", 0, 120), ChunkSpan("a1", "d1", 90, 300)]})
    chunking_b = ChunkIndex(
        {"d1": [ChunkSpan("b0", "d1", 0, 160), ChunkSpan("b1", "d1", 160, 400)]}
    )
    assert chunking_a.relevant_chunks(it) == {"a1"}  # a0 covers only 20% of the span
    assert chunking_b.relevant_chunks(it) == {"b0"}  # 60% vs 40%: same item, new chunks
    assert ChunkIndex({}).evidence_chunks(it) == [set()]  # unmapped span


# ---------------------------------------------------------------- validators
def test_sentence_spans_respect_hard_wraps_and_keep_code_whole():
    text = (
        "The kubelet restarts\nfailed containers. It waits first.\n\n"
        "```yaml\nkind: Pod\n\nspec: {}\n```\n"
    )
    spans = sentence_spans(text, 0, len(text))
    parts = [text[s:e] for s, e in spans]
    assert parts[0] == "The kubelet restarts\nfailed containers."  # wrap is not a boundary
    assert parts[1] == "It waits first."
    assert parts[2].startswith("```yaml") and parts[2].endswith("```")


def test_answer_support_requires_new_answer_words_from_evidence():
    evidence = "CoreDNS currently caches for 30 seconds."
    assert V.check_answer_supported(
        "The default termination grace period for a Pod is 30 seconds", "", evidence
    )  # the fabricated fact from the pilot
    assert (
        V.check_answer_supported(
            "It caches for 30 seconds.", "How long does CoreDNS cache?", evidence
        )
        is None
    )
    assert {V.stem(w) for w in ("require", "requires", "required", "requiring")} == {"requir"}
    assert V.stem("caches") == V.stem("cache") and V.stem("policies") == "policy"


def test_paraphrase_uses_containment_not_jaccard():
    evidence = (
        "Typical operations on volumes are supported assuming that the driver supports them " * 3
    )
    copied = "What typical operations are supported on volumes?"
    assert V.check_paraphrase(copied, evidence) == "not_paraphrased"
    assert (
        V.check_paraphrase("Which storage actions work for per-pod scratch space?", evidence)
        is None
    )
    # generic words ("Kubernetes cluster") no longer dilute the ratio (pilot 2 miss)
    assert (
        V.check_paraphrase(
            "What command is used to delete a Service in a Kubernetes cluster?",
            "To delete the Service, enter this command",
        )
        == "not_paraphrased"
    )


def test_self_contained_rejects_passage_references():
    assert V.check_self_contained("How many pods need eviction according to the status message?")
    assert V.check_self_contained("What does the example output show?")
    assert V.check_self_contained(
        "How does the concept of Workload differ between the two passages?"
    )
    assert V.check_self_contained("What is the default grace period for a Pod?") is None


def test_uses_both_rejects_single_source_answers():
    a = "DaemonSet pod templates require restartPolicy Always."
    b = "Topology manager policies control NUMA alignment on nodes."
    assert V.check_uses_both("DaemonSets require restartPolicy Always.", a, b)
    assert (
        V.check_uses_both(
            "DaemonSet templates need restartPolicy Always, and topology manager "
            "policies handle NUMA alignment.",
            a,
            b,
        )
        is None
    )


def test_perturb_number_is_deterministic():
    ev = "By default, all deletes are graceful within 30 seconds."
    assert V.perturb_number("The default grace period is 30 seconds", ev) == (
        "The default grace period is 60 seconds",
        "30",
        "60",
    )
    assert V.perturb_number("It retries 1 time", "It retries 1 time.")[2] == "3"  # 0/1 -> +2
    assert V.perturb_number("No numbers here", ev) is None


def test_perturb_number_never_uses_a_value_present_in_evidence():
    ev = "Deletes are graceful within 30 seconds; forced after 60 seconds."
    assert V.perturb_number("The default grace period is 30 seconds", ev) is None  # 60 taken


def test_link_only_evidence_rejected():
    assert V.check_substantive_quote("See [Topology Manager](/docs/tasks/x/) for more details.")
    assert V.check_substantive_quote("The scheduler assigns Pods to Nodes.") is None


def test_duplicate_guard():
    g = V.DuplicateGuard()
    g.add("What is the default termination grace period for a Pod?")
    assert g.check("What's the default termination grace period of a Pod?") == "near_duplicate"
    assert g.check("How do I roll back a Deployment?") is None


# ---------------------------------------------------------------- splits
def test_splits_are_stratified_and_reproducible():
    def make():
        items = [item(i, QuestionType.SIMPLE_FACTUAL) for i in range(10)]
        items += [item(100 + i, QuestionType.NUMERICAL) for i in range(5)]
        assign_splits(items, 0.3, seed=7)
        return {i.id: i.split for i in items}

    first = make()
    assert first == make()
    dev = [k for k, v in first.items() if v == "dev"]
    assert sum(k.startswith("q1") and len(k) == 4 for k in dev) == 2  # round(5 * 0.3) numerical
    assert len(dev) == 3 + 2


# ---------------------------------------------------------------- runner
class ListRetriever:
    name = "list"

    def __init__(self, rankings):
        self.rankings = rankings

    def retrieve(self, query, k, filters=None):
        ids = self.rankings.get(query, [])
        return [
            RetrievalResult(
                document_id=c.split(":")[0],
                chunk_id=c,
                text="",
                score=1.0,
                rank=r,
                retriever="list",
                start_char=0,
                end_char=1,
            )
            for r, c in enumerate(ids[:k], start=1)
        ]


def test_runner_scores_by_type_and_excludes_unanswerable():
    q0 = item(0, evidence=[span("d1", 0, 50)])
    q1 = item(1, QuestionType.MULTI_HOP, evidence=[span("d1", 0, 50), span("d2", 0, 50)])
    q2 = item(2, QuestionType.UNANSWERABLE, evidence=[])
    index = ChunkIndex(
        {"d1": [ChunkSpan("d1:c0", "d1", 0, 60)], "d2": [ChunkSpan("d2:c0", "d2", 0, 60)]}
    )
    retriever = ListRetriever({q0.question: ["d9:c0", "d1:c0"], q1.question: ["d1:c0", "d8:c0"]})
    report, rows = evaluate_retrieval(
        retriever, [q0, q1, q2], index, dataset="t", dataset_version="v", split="test", ks=[1, 2]
    )
    assert report.n_evaluated == 2 and report.n_excluded_no_evidence == 1
    by_id = {r.item_id: r.metrics for r in rows}
    assert by_id["q0"]["mrr"] == 0.5 and by_id["q0"]["hit@1"] == 0.0
    assert by_id["q1"]["evidence_recall@2"] == 0.5 and by_id["q1"]["doc_recall@2"] == 0.5
    assert report.by_type["multi_hop"]["evidence_recall@2"] == 0.5
    assert report.overall["mrr"] == pytest.approx(0.75)
