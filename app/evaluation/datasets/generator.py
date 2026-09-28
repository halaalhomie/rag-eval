"""Synthetic evaluation-set generation over the ingested corpus.

For each question type: sample source chunk(s) with a seeded RNG, ask the LLM for a
question, answer and verbatim evidence quote, then run deterministic validators. Only
items that pass every validator are kept. Rejection reasons are counted per type and
published in the dataset metadata, so the filtering is itself inspectable.

Sampling rules keep the set diverse: each chunk is used at most once, and each document
yields at most MAX_PER_DOCUMENT items.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from sqlalchemy import func, select

from app.db.models import Chunk, Document
from app.evaluation.datasets import prompts as P
from app.evaluation.datasets import validators as V
from app.evaluation.datasets.labels import ChunkIndex
from app.evaluation.datasets.schema import EvalItem, EvidenceSpan, Provenance, QuestionType
from app.generation.llm import ChatMessage, LLMClient, LLMConnectionError, LLMError
from app.generation.structured import complete_structured
from app.ingestion.chunking import sentence_spans
from app.retrieval.base import Retriever
from app.retrieval.dense import SessionFactory

logger = logging.getLogger(__name__)

MAX_PER_DOCUMENT = 3
MIN_TOKENS = 80
UNANSWERABLE_GROUND_TRUTH = (
    "Not answerable from the Kubernetes documentation corpus; the system should say it "
    "does not know rather than answer."
)


def nonexistent_ground_truth(term: str) -> str:
    """A fake feature can sit next to a real one (a fake `--force-delete` flag vs the
    documented `--force`), so a flat "I don't know" is not the only correct response:
    saying the feature is undocumented and pointing to the real alternative is too."""
    return (
        f"The Kubernetes documentation corpus does not document `{term}`. A correct "
        "response says so (and may point to a related documented feature) rather than "
        "explaining how to use it."
    )


# Technologies checked to be absent from the corpus before use (see _ood_topics).
OOD_CANDIDATES = (
    "Apache Kafka consumer groups",
    "PostgreSQL query planning",
    "Terraform state files",
    "React hooks",
    "Django migrations",
    "Apache Spark shuffles",
    "Ansible playbooks",
    "RabbitMQ exchanges",
    "Elasticsearch shard allocation",
    "TensorFlow training loops",
    "Rust borrow checking",
    "Git rebase workflows",
    "Cassandra compaction",
    "Jenkins pipelines",
    "MongoDB replica sets",
    "Airflow DAG scheduling",
    "Snowflake warehouses",
    "Kotlin coroutines",
    "GraphQL resolvers",
    "Flutter widgets",
)
DIFFICULTY = {
    QuestionType.SIMPLE_FACTUAL: "easy",
    QuestionType.KEYWORD_HEAVY: "easy",
    QuestionType.NUMERICAL: "easy",
    QuestionType.SEMANTIC: "medium",
    QuestionType.AMBIGUOUS: "medium",
    QuestionType.COMPARISON: "medium",
    QuestionType.UNANSWERABLE: "medium",
    QuestionType.MULTI_HOP: "hard",
    QuestionType.ADVERSARIAL: "hard",
}
_NUMERIC = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:seconds?|minutes?|hours?|days?|ms|MiB|GiB|Mi|Gi|KiB|%|percent|"
    r"bytes|nodes|pods|replicas|characters)\b"
)
_IDENT = re.compile(r"`([A-Za-z-][\w.\-/=]{3,40})`")
_LINK = re.compile(r"\]\((/docs/[a-z0-9/_-]+?)/?(?:#[^)]*)?\)")
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z]+\b")
_QUESTION_WORDS = frozenset(
    "What How When Why Which Does Is Can Kubernetes If In Do Are Who Where Should I A The".split()
)


class Rejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Src:
    id: str
    document_id: str
    text: str
    start: int
    end: int
    section: str | None
    title: str
    url: str | None
    category: str | None
    ordinal: int
    source: str


@dataclass
class TypeStats:
    attempts: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, int]:
        return {"attempts": self.attempts, "accepted": self.accepted, **self.rejected}


class DatasetGenerator:
    def __init__(
        self,
        llm: LLMClient,
        model: str,
        session_factory: SessionFactory,
        retriever: Retriever,
        seed: int = 42,
    ):
        self.llm = llm
        self.model = model
        self.session_factory = session_factory
        self.retriever = retriever
        self.rng = random.Random(seed)
        self.dupes = V.DuplicateGuard()
        self.used_chunks: set[str] = set()
        self.per_doc: Counter[str] = Counter()
        self.stats: dict[str, TypeStats] = defaultdict(TypeStats)
        self._consecutive_llm_errors = 0
        self._sentence_cache: dict[tuple[str, bool], list[tuple[int, int]]] = {}
        self._load()

    # ---------------------------------------------------------------- corpus access
    def _load(self) -> None:
        with self.session_factory() as s:
            docs = s.execute(
                select(Document.id, Document.content, Document.source, Document.name)
            ).all()
            self.content = {d.id: d.content for d in docs}
            self.doc_source = {d.id: d.source for d in docs}
            all_chunks = [
                (self._src(c), c.token_count)
                for c in s.execute(select(Chunk).order_by(Chunk.id)).scalars()
            ]
            self.index = ChunkIndex.load(s)
        self.chunks = [
            c
            for c, tokens in all_chunks
            if tokens >= MIN_TOKENS
            and _prose_fraction(c.text) >= 0.5
            and _html_fraction(c.text) < 0.05
        ]
        self.by_id = {c.id: c for c in self.chunks}
        # Intro chunk of each concept page: describes what the object *is* (comparisons).
        self.intro = {
            c.document_id: c
            for c, _ in all_chunks
            if c.ordinal == 0 and c.category == "concepts" and len(c.text) > 200
        }
        self.url_to_doc = {c.url: c.document_id for c in self.chunks if c.url}
        logger.info("Eligible chunks: %d", len(self.chunks))

    def _src(self, c: Chunk) -> Src:
        return Src(
            id=c.id,
            document_id=c.document_id,
            text=c.text,
            start=c.start_char,
            end=c.end_char,
            section=c.section,
            title=c.metadata_.get("title", ""),
            url=c.metadata_.get("url"),
            category=c.metadata_.get("doc_category"),
            ordinal=c.ordinal,
            source=self.doc_source[c.document_id],
        )

    def term_in_corpus(self, term: str) -> bool:
        escaped = term.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        with self.session_factory() as s:
            n = s.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.content.ilike(f"%{escaped}%"))
            )
        return bool(n)

    def _available(self, src: Src) -> bool:
        return src.id not in self.used_chunks and self.per_doc[src.document_id] < MAX_PER_DOCUMENT

    def _sentences(self, src: Src, prose_only: bool = False) -> list[tuple[int, int]]:
        """Numbered units of a chunk. `prose_only` drops code blocks: used for two-source
        questions, where one big YAML "sentence" would satisfy the uses-both-sources check
        by sheer word count (seen in pilot 3)."""
        key = (src.id, prose_only)
        if key not in self._sentence_cache:
            text = self.content[src.document_id]
            spans = sentence_spans(text, src.start, src.end)
            if prose_only:
                spans = [(s, e) for s, e in spans if not text[s:e].lstrip().startswith("```")]
            self._sentence_cache[key] = spans
        return self._sentence_cache[key]

    def _passage(self, src: Src, prose_only: bool = False) -> str:
        text = self.content[src.document_id]
        sents = self._sentences(src, prose_only)
        return P.numbered(src.title, src.section, [text[s:e] for s, e in sents])

    def _spans(
        self, src: Src, sentence_ids: list[int], prose_only: bool = False
    ) -> list[EvidenceSpan]:
        """Map 1-based sentence numbers to exact evidence spans (contiguous runs merged)."""
        sents = self._sentences(src, prose_only)
        ids = sorted(set(sentence_ids))
        if not ids or ids[0] < 1 or ids[-1] > len(sents):
            raise Rejected("invalid_sentence_ids")
        runs: list[list[int]] = [[ids[0]]]
        for i in ids[1:]:
            if i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        text = self.content[src.document_id]
        spans = []
        for run in runs:
            s, e = sents[run[0] - 1][0], sents[run[-1] - 1][1]
            spans.append(
                EvidenceSpan(document_id=src.document_id, start_char=s, end_char=e, quote=text[s:e])
            )
        return spans

    def _ask(self, prompt: str, schema):
        try:
            out, _ = complete_structured(
                self.llm,
                [ChatMessage(role="user", content=prompt)],
                schema,
                model=self.model,
                max_tokens=400,
            )
        except LLMConnectionError:
            raise
        except LLMError as exc:
            self._consecutive_llm_errors += 1
            if self._consecutive_llm_errors >= 10:
                raise RuntimeError("10 consecutive LLM failures; aborting") from exc
            raise Rejected("malformed_output") from exc
        self._consecutive_llm_errors = 0
        return out

    # ---------------------------------------------------------------- item assembly
    def _item(
        self,
        qtype: QuestionType,
        question: str,
        answer: str,
        evidence: list[EvidenceSpan],
        srcs: list[Src],
        checks: list[str],
        subtype: str | None = None,
        notes: dict[str, str] | None = None,
        check_support: bool = True,
    ) -> EvalItem:
        for reason in (
            V.check_question(question),
            V.check_self_contained(question),
            self.dupes.check(question),
        ):
            if reason:
                raise Rejected(reason)
        checks = ["question_well_formed", "self_contained", "not_duplicate", *checks]
        if evidence:
            evidence_text = " ".join(e.quote for e in evidence)
            for reason in (
                V.check_answer(answer),
                V.check_self_contained(answer),
                V.check_answer_supported(answer, question, evidence_text)
                if check_support
                else None,
            ):
                if reason:
                    raise Rejected(reason)
            checks += ["answer_non_empty", "evidence_from_sentence_ids"]
            if check_support:
                checks.append("answer_supported_by_evidence")
        docs = sorted({e.document_id for e in evidence})
        item = EvalItem(
            id="q_" + hashlib.sha1(question.encode()).hexdigest()[:10],
            question=question.strip(),
            ground_truth_answer=answer.strip(),
            question_type=qtype,
            difficulty=DIFFICULTY[qtype],
            split="test",  # assigned later by assign_splits
            evidence=evidence,
            relevant_document_ids=docs,
            subtype=subtype,
            notes=notes or {},
            provenance=Provenance(
                method="synthetic_llm",
                generator_model=self.model,
                prompt_version=P.PROMPT_VERSION,
                validators_passed=checks,
            ),
        )
        item.relevant_chunk_ids = sorted(self.index.relevant_chunks(item))
        if evidence and not item.relevant_chunk_ids:
            raise Rejected("evidence_not_in_any_chunk")
        self.dupes.add(question)
        for src in srcs:
            self.used_chunks.add(src.id)
            self.per_doc[src.document_id] += 1
        return item

    # ---------------------------------------------------------------- per-type builders
    def _single(self, qtype: QuestionType, pool: list[Src]) -> Callable[[], EvalItem]:
        def build() -> EvalItem:
            src = self._pick(pool)
            term = self._identifier(src) if qtype is QuestionType.KEYWORD_HEAVY else None
            out = self._ask(P.single(qtype.value, self._passage(src), term), P.SingleOut)
            spans = self._spans(src, out.evidence_sentences)
            evidence = " ".join(s.quote for s in spans)
            checks: list[str] = []
            if qtype is not QuestionType.KEYWORD_HEAVY:
                if reason := V.check_no_leakage(out.question, evidence):
                    raise Rejected(reason)
                checks.append("no_leakage")
            if qtype is QuestionType.SEMANTIC:
                if reason := V.check_paraphrase(out.question, evidence):
                    raise Rejected(reason)
                checks.append("paraphrased")
            if qtype is QuestionType.KEYWORD_HEAVY:
                for text in (out.question, evidence):
                    if reason := V.check_contains_term(text, term):
                        raise Rejected(reason)
                checks.append("exact_term_in_question_and_evidence")
            if qtype is QuestionType.NUMERICAL:
                if reason := V.check_numeric_answer(out.answer, evidence):
                    raise Rejected(reason)
                checks.append("number_from_evidence")
            notes = {"required_term": term} if term else None
            return self._item(qtype, out.question, out.answer, spans, [src], checks, notes=notes)

        return build

    def _pair(self, qtype: QuestionType) -> Callable[[], EvalItem]:
        def build() -> EvalItem:
            a, b = self._link_pair() if qtype is QuestionType.MULTI_HOP else self._sibling_pair()
            prompt = P.pair(qtype.value, self._passage(a, True), self._passage(b, True))
            out = self._ask(prompt, P.PairOut)
            spans_a = self._spans(a, out.evidence_sentences_a, prose_only=True)
            spans_b = self._spans(b, out.evidence_sentences_b, prose_only=True)
            text_a = " ".join(s.quote for s in spans_a)
            text_b = " ".join(s.quote for s in spans_b)
            for reason in (
                V.check_no_leakage(out.question, text_a),
                V.check_no_leakage(out.question, text_b),
                V.check_substantive_quote(text_a),
                V.check_substantive_quote(text_b),
                V.check_uses_both(out.answer, text_a, text_b),
            ):
                if reason:
                    raise Rejected(reason)
            return self._item(
                qtype,
                out.question,
                out.answer,
                spans_a + spans_b,
                [a, b],
                [
                    "no_leakage",
                    "substantive_evidence",
                    "answer_uses_both_sources",
                    "two_source_documents",
                ],
                subtype="linked_documents"
                if qtype is QuestionType.MULTI_HOP
                else "similar_concepts",
            )

        return build

    def _ambiguous(self) -> EvalItem:
        src = self._pick(self.chunks)
        out = self._ask(P.ambiguous(self._passage(src)), P.AmbiguousOut)
        spans = self._spans(src, out.evidence_sentences)
        if reason := V.check_question(out.specific_question):
            raise Rejected(f"specific_{reason}")
        subjects = [
            w for w in _CAPITALIZED.findall(out.specific_question) if w not in _QUESTION_WORDS
        ]
        if not subjects:
            raise Rejected("no_subject_to_remove")
        if reason := V.check_subject_removed(out.ambiguous_question, [src.title, *subjects]):
            raise Rejected(reason)
        return self._item(
            QuestionType.AMBIGUOUS,
            out.ambiguous_question,
            out.answer,
            spans,
            [src],
            ["subject_removed"],
            notes={"disambiguated_question": out.specific_question.strip()},
        )

    def _adversarial(self, pool: list[Src]) -> EvalItem:
        """False premise built by code, not by the LLM (in the pilot, 3/3 LLM-written
        premises were wrong). The LLM states a TRUE numeric fact, which must be supported by
        the selected sentences and share their number; code then changes that number to
        create the false premise. The ground truth is the verified true statement."""
        src = self._pick(pool)
        out = self._ask(P.numeric_fact(self._passage(src)), P.FactOut)
        spans = self._spans(src, out.evidence_sentences)
        evidence = " ".join(s.quote for s in spans)
        fact = " ".join(out.fact_statement.split()).rstrip(".")
        for reason in (
            V.check_numeric_answer(fact, evidence),
            V.check_self_contained(fact),
            # stricter than for answers: the whole fact must come from the evidence
            V.check_answer_supported(fact, "", evidence, min_fraction=0.7),
        ):
            if reason:
                raise Rejected(reason)
        perturbed = V.perturb_number(fact, evidence)
        if perturbed is None:
            raise Rejected("nothing_to_perturb")
        false_statement, original, replacement = perturbed
        question = f"I read that {false_statement[0].lower() + false_statement[1:]}. Why is that?"
        return self._item(
            QuestionType.ADVERSARIAL,
            question,
            f"That premise is incorrect: {fact}.",
            spans,
            [src],
            ["number_from_evidence", "fact_supported_by_evidence", "premise_perturbed_by_code"],
            subtype="perturbed_number",
            notes={
                "false_premise": false_statement,
                "true_value": original,
                "false_value": replacement,
            },
            check_support=False,  # the answer restates the fact, which is checked above
        )

    def _nonexistent(self) -> EvalItem:
        src = self._pick(self.chunks, mark_used=False)
        out = self._ask(P.fake_feature(self._passage(src)), P.FakeFeatureOut)
        term = out.fake_term.strip().strip("`")
        if len(term) < 5:
            raise Rejected("fake_term_too_short")
        if reason := V.check_contains_term(out.question, term):
            raise Rejected(reason)
        if self.term_in_corpus(term):
            raise Rejected("fake_term_exists_in_corpus")
        return self._item(
            QuestionType.UNANSWERABLE,
            out.question,
            nonexistent_ground_truth(term),
            [],
            [],
            ["term_absent_from_corpus"],
            subtype="nonexistent_feature",
            notes={"fake_term": term, "topic_source": src.document_id},
        )

    def _out_of_domain(self, topics: list[str]) -> EvalItem:
        topic, angle = self.rng.choice(topics), self.rng.choice(P.OOD_ANGLES)
        out = self._ask(P.out_of_domain(topic, angle), P.OutOfDomainOut)
        if reason := V.check_contains_term(out.question, _ood_key(topic)):
            raise Rejected(reason)
        if re.search(r"kubernetes|kubectl|\bpods?\b|container", out.question, re.I):
            raise Rejected("mentions_kubernetes")
        return self._item(
            QuestionType.UNANSWERABLE,
            out.question,
            UNANSWERABLE_GROUND_TRUTH,
            [],
            [],
            ["topic_absent_from_corpus", "no_kubernetes_terms"],
            subtype="out_of_domain",
            notes={"topic": topic, "angle": angle},
        )

    def seed_existing(self, items: list[EvalItem]) -> None:
        """Extend mode: existing questions count as duplicates, and their source chunks and
        documents count as used, so a top-up run never repeats them."""
        for item in items:
            self.dupes.add(item.question)
            for chunk_id in self.index.relevant_chunks(item):
                self.used_chunks.add(chunk_id)
            for doc in item.relevant_document_ids:
                self.per_doc[doc] += 1

    # ---------------------------------------------------------------- sampling
    def _pick(self, pool: list[Src], mark_used: bool = True) -> Src:
        candidates = [c for c in pool if self._available(c)] if mark_used else pool
        if not candidates:
            raise Rejected("pool_exhausted")
        return self.rng.choice(candidates)

    def _identifier(self, src: Src) -> str:
        # Only identifier-like terms (flags, dotted fields, camelCase): a plain word such
        # as "Deployment" does not make a keyword query.
        idents = [i for i in _IDENT.findall(src.text) if re.search(r"[-.=/]|[a-z][A-Z]", i)]
        if not idents:
            raise Rejected("no_identifier")
        return self.rng.choice(idents)

    def _link_pair(self) -> tuple[Src, Src]:
        candidates = [
            c
            for c in self.chunks
            if self._available(c) and c.category in {"concepts", "tasks"} and _LINK.search(c.text)
        ]
        self.rng.shuffle(candidates)
        for a in candidates[:50]:
            for m in _LINK.finditer(a.text):
                url = f"https://kubernetes.io{m.group(1)}/"
                target = self.url_to_doc.get(url)
                if (
                    not target
                    or target == a.document_id
                    or self.per_doc[target] >= MAX_PER_DOCUMENT
                ):
                    continue
                # Best chunk of the linked page for the text around the link.
                ctx = a.text[max(0, m.start() - 300) : m.end() + 100]
                for h in self.retriever.retrieve(ctx, 3, filters={"url": url}):
                    b = self.by_id.get(h.chunk_id)
                    if b and self._available(b):
                        return a, b
        raise Rejected("no_link_pair")

    def _sibling_pair(self) -> tuple[Src, Src]:
        """Two *comparable* concept pages: A's intro plus the most similar other intro in
        the same docs section (e.g. Deployment vs StatefulSet). In the pilot, random
        siblings (metrics vs version emulation) produced meaningless comparisons."""
        intros = [c for c in self.intro.values() if self._available(c)]
        self.rng.shuffle(intros)
        by_id = {c.id: c for c in intros}
        for a in intros[:20]:
            section = PurePosixPath(a.source).parent
            for h in self.retriever.retrieve(a.text, 30, filters={"doc_category": "concepts"}):
                b = by_id.get(h.chunk_id)
                if (
                    b
                    and b.document_id != a.document_id
                    and PurePosixPath(b.source).parent == section
                ):
                    return a, b
        raise Rejected("no_sibling_pair")

    def _ood_topics(self) -> list[str]:
        topics = [t for t in OOD_CANDIDATES if not self.term_in_corpus(_ood_key(t))]
        logger.info(
            "Out-of-domain topics absent from corpus: %d/%d", len(topics), len(OOD_CANDIDATES)
        )
        return topics

    # ---------------------------------------------------------------- driver
    def generate(self, quotas: dict[str, int], max_attempt_factor: int = 6) -> list[EvalItem]:
        numeric = [c for c in self.chunks if _NUMERIC.search(c.text)]
        with_idents = [c for c in self.chunks if _IDENT.search(c.text)]
        ood = self._ood_topics()
        builders: dict[str, Callable[[], EvalItem]] = {
            "simple_factual": self._single(QuestionType.SIMPLE_FACTUAL, self.chunks),
            "semantic": self._single(QuestionType.SEMANTIC, self.chunks),
            "keyword_heavy": self._single(QuestionType.KEYWORD_HEAVY, with_idents),
            "numerical": self._single(QuestionType.NUMERICAL, numeric),
            "multi_hop": self._pair(QuestionType.MULTI_HOP),
            "comparison": self._pair(QuestionType.COMPARISON),
            "ambiguous": self._ambiguous,
            "adversarial": lambda: self._adversarial(numeric),
            "unanswerable/nonexistent_feature": self._nonexistent,
            "unanswerable/out_of_domain": lambda: self._out_of_domain(ood),
        }
        items: list[EvalItem] = []
        for key, quota in quotas.items():
            stats = self.stats[key]
            while stats.accepted < quota and stats.attempts < quota * max_attempt_factor:
                stats.attempts += 1
                try:
                    items.append(builders[key]())
                    stats.accepted += 1
                except Rejected as r:
                    stats.rejected[r.reason] += 1
                    if r.reason in {"pool_exhausted", "no_link_pair", "no_sibling_pair"}:
                        break
                except ValueError as exc:  # schema-level validation of the assembled item
                    stats.rejected[f"invalid_item:{type(exc).__name__}"] += 1
            logger.info(
                "%s: %d/%d accepted in %d attempts %s",
                key,
                stats.accepted,
                quota,
                stats.attempts,
                dict(stats.rejected),
            )
        return items


def _ood_key(topic: str) -> str:
    parts = topic.split()
    return parts[1] if parts[0] == "Apache" else parts[0]


def _html_fraction(text: str) -> float:
    """Share of characters inside HTML tags (some reference pages embed raw tables)."""
    return sum(len(m.group()) for m in re.finditer(r"<[^<>]{1,200}>", text)) / max(len(text), 1)


def _prose_fraction(text: str) -> float:
    """Share of characters outside fenced code blocks."""
    code = sum(len(m.group()) for m in re.finditer(r"```.*?(```|$)", text, re.DOTALL))
    return 1 - code / max(len(text), 1)


def assign_splits(items: list[EvalItem], dev_fraction: float, seed: int) -> None:
    """Stratified by question type: thresholds are tuned on dev, results reported on test."""
    rng = random.Random(seed)
    by_type: dict[QuestionType, list[EvalItem]] = defaultdict(list)
    for item in items:
        by_type[item.question_type].append(item)
    for group in by_type.values():
        group.sort(key=lambda i: i.id)
        rng.shuffle(group)
        n_dev = round(len(group) * dev_fraction)
        for i, item in enumerate(group):
            item.split = "dev" if i < n_dev else "test"
