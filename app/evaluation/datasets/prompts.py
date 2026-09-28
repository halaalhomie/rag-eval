"""Prompts and output schemas for synthetic question generation.

Kept short and literal: the generator is a 3B model. Passages are shown as numbered
sentences, and the model returns the *numbers* of the sentences that contain the answer.
Evidence spans are then mapped deterministically from sentence offsets, so they are
always exact.

(qgen-v1 asked for verbatim quotes. The 3B model paraphrased them often enough that most
multi-hop attempts failed; see docs/evaluation.md.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROMPT_VERSION = "qgen-v4"

_COMMON = (
    "You write evaluation questions for a question-answering system over the Kubernetes "
    "documentation. Questions must sound like a real engineer asking, make sense on their "
    "own (never mention 'the passage' or 'the example'), and have a short factual answer "
    "(1-2 sentences) that is stated in the passage."
)
_EVIDENCE = "evidence_sentences lists the numbers of the 1-3 sentences that contain the answer."

Sentences = Field(min_length=1, max_length=3)


class SingleOut(BaseModel):
    question: str
    answer: str
    evidence_sentences: list[int] = Sentences


class PairOut(BaseModel):
    question: str
    answer: str
    evidence_sentences_a: list[int] = Sentences
    evidence_sentences_b: list[int] = Sentences


class AmbiguousOut(BaseModel):
    specific_question: str
    ambiguous_question: str
    answer: str
    evidence_sentences: list[int] = Sentences


class FactOut(BaseModel):
    fact_statement: str
    evidence_sentences: list[int] = Sentences


class FakeFeatureOut(BaseModel):
    fake_term: str
    question: str


class OutOfDomainOut(BaseModel):
    question: str


def numbered(title: str, section: str | None, sentences: list[str]) -> str:
    header = " > ".join(p for p in (title, section) if p)
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, start=1))
    return f"({header})\n{body}"


def single(kind: str, psg: str, term: str | None = None) -> str:
    extra = {
        "simple_factual": "Ask a direct factual question about one specific detail.",
        "semantic": (
            "Ask about the underlying idea using DIFFERENT words from the passage: "
            "paraphrase, and do not reuse its key terms."
        ),
        "keyword_heavy": f"The question MUST contain the exact term `{term}`.",
        "numerical": (
            "Ask a question whose answer is a specific number or quantity stated in the "
            "passage. The answer must include that number."
        ),
    }[kind]
    return f"{_COMMON}\n\n{extra} {_EVIDENCE}\n\nPassage:\n{psg}"


def pair(kind: str, psg_a: str, psg_b: str) -> str:
    extra = {
        "multi_hop": (
            "Write ONE question that can only be answered by combining a fact from "
            "passage A with a fact from passage B. The answer must use both facts."
        ),
        "comparison": (
            "Write ONE question that compares the two things described in passage A and "
            "passage B (for example, how they differ or when to use each). Name both "
            "things in the question."
        ),
    }[kind]
    return (
        f"{_COMMON}\n\n{extra} Phrase it as a question ending with '?'. "
        "evidence_sentences_a lists sentence numbers from passage A, "
        f"evidence_sentences_b from passage B.\n\nPassage A:\n{psg_a}\n\nPassage B:\n{psg_b}"
    )


def ambiguous(psg: str) -> str:
    return (
        f"{_COMMON}\n\nFirst write specific_question, a clear factual question about the "
        "passage. Then write ambiguous_question: the same question made vague, replacing "
        "the main Kubernetes object or feature with 'it', 'this' or 'that thing', the way "
        "a user might ask without context (for example 'What happens when it crashes?'). "
        f"{_EVIDENCE}\n\nPassage:\n{psg}"
    )


def numeric_fact(psg: str) -> str:
    """Used to build false-premise items. The 3B model cannot write false premises
    reliably, so it only states a TRUE fact and code perturbs it (see generator). There is
    deliberately no example sentence: in the pilot the model copied the example verbatim."""
    return (
        "State ONE fact from the passage that contains a specific number, as a single "
        "self-contained sentence naming what the number refers to. Use the passage's own "
        "words. evidence_sentences lists the numbers of the sentences stating that fact."
        f"\n\nPassage:\n{psg}"
    )


def fake_feature(psg: str) -> str:
    return (
        "Invent a plausible-sounding but NON-EXISTENT Kubernetes feature, flag or field, "
        "related to the topic of the passage below. fake_term is its exact name (for "
        "example a flag like --auto-heal-pods or a field like spec.selfRepair). question "
        f"asks how to use it.\n\nPassage (topic only):\n{psg}"
    )


# At temperature 0 the same topic prompt always yields the same question, so the first
# full run produced 81 near-duplicates for 9 topics. An angle varies the prompt.
OOD_ANGLES = (
    "configuration",
    "debugging an error",
    "performance tuning",
    "security",
    "best practices",
    "upgrading versions",
)


def out_of_domain(topic: str, angle: str) -> str:
    return (
        f"Write one specific technical question an engineer might ask about {topic}, "
        f"focused on {angle}. Do not mention Kubernetes, containers or pods."
    )
