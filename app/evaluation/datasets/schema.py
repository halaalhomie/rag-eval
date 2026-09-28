"""Evaluation dataset schema and JSONL I/O.

Relevance is anchored to **evidence spans** (character offsets into `documents.content`),
not to chunk IDs. `relevant_chunk_ids` is stored for convenience, as a snapshot under the
chunking used at generation time, but evaluation always recomputes chunk relevance from
the spans against the *current* index (see labels.py). Re-chunking therefore never
invalidates the dataset.

Every item records how it was produced. Synthetic items must never be presented as
human-annotated.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QuestionType(StrEnum):
    SIMPLE_FACTUAL = "simple_factual"
    SEMANTIC = "semantic"  # paraphrased: little lexical overlap with the evidence
    KEYWORD_HEAVY = "keyword_heavy"  # hinges on an exact identifier (flag, field, gate)
    MULTI_HOP = "multi_hop"  # needs evidence from two linked documents
    COMPARISON = "comparison"  # contrasts two documented objects
    NUMERICAL = "numerical"  # answer is a documented number / quantity
    AMBIGUOUS = "ambiguous"  # underspecified subject; tests query rewriting
    UNANSWERABLE = "unanswerable"  # corpus does not contain the answer
    ADVERSARIAL = "adversarial"  # false premise contradicted by the corpus


# Question types with no supporting evidence in the corpus: excluded from retrieval metrics.
NO_EVIDENCE_TYPES = frozenset({QuestionType.UNANSWERABLE})


class EvidenceSpan(BaseModel):
    document_id: str
    start_char: int = Field(ge=0)
    end_char: int
    quote: str  # exact `documents.content[start_char:end_char]`

    @model_validator(mode="after")
    def _check(self) -> EvidenceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("evidence span must be non-empty")
        if len(self.quote) != self.end_char - self.start_char:
            raise ValueError("quote length does not match span")
        return self


class Provenance(BaseModel):
    method: Literal["synthetic_llm", "human", "template"]
    generator_model: str | None = None
    prompt_version: str | None = None
    # Deterministic validators that passed; see validators.py.
    validators_passed: list[str] = Field(default_factory=list)
    reviewed_by: str | None = None  # set only when a person actually reviewed the item


class EvalItem(BaseModel):
    id: str
    question: str
    ground_truth_answer: str
    question_type: QuestionType
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["dev", "test"]
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)  # snapshot, see module doc
    subtype: str | None = None  # e.g. "out_of_domain", "nonexistent_feature"
    notes: dict[str, str] = Field(default_factory=dict)  # e.g. disambiguated question
    provenance: Provenance

    @model_validator(mode="after")
    def _check(self) -> EvalItem:
        if self.question_type in NO_EVIDENCE_TYPES:
            if self.evidence:
                raise ValueError(f"{self.question_type} items must not have evidence")
        elif not self.evidence:
            raise ValueError(f"{self.question_type} items require evidence spans")
        docs = {e.document_id for e in self.evidence}
        if set(self.relevant_document_ids) != docs:
            raise ValueError("relevant_document_ids must equal the evidence documents")
        return self

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


class DatasetMeta(BaseModel):
    name: str
    version: str  # content hash of the JSONL file
    created_at: str
    corpus: str
    corpus_commit: str | None
    chunker_fingerprint: str | None
    generator_model: str | None
    seed: int
    counts: dict[str, int]
    split_counts: dict[str, int]
    generation_stats: dict[str, dict[str, int]] = Field(default_factory=dict)
    runs: list[dict[str, str | int]] = Field(default_factory=list)  # one per generation run
    synthetic: bool = True
    description: str = ""


def dataset_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def save_dataset(items: list[EvalItem], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_none=True) + "\n")
    return dataset_version(path)


def load_dataset(
    path: Path, split: str | None = None, types: set[QuestionType] | None = None
) -> list[EvalItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = EvalItem.model_validate(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: invalid eval item: {exc}") from exc
            if (split is None or item.split == split) and (
                types is None or item.question_type in types
            ):
                items.append(item)
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items
