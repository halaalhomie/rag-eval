"""Deterministic validators for synthetic eval items.

The generator is a small local LLM, so nothing it produces is trusted without checks. Each
validator is a pure function returning None (pass) or a short rejection reason. Accepted
items record which validators they passed; rejection reasons are counted per question type
and published in the dataset metadata.

What these checks cannot establish: that a question is *natural*, or that the answer is
the *best* answer. Those need human review; see docs/evaluation.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_WORD = re.compile(r"\w+")
# Words that say nothing about *what* is asked in this corpus.
_GENERIC = frozenset("kubernetes k8s cluster clusters use used using".split())
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been by with as at from that this "
    "it its can do does how what which when why who you your if not no into than then there "
    "their they them these those will would should could about over under".split()
)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in _STOP and (len(w) > 2 or w.isdigit())}


def stem(word: str) -> str:
    """Very light stemming: require/requires/required/requiring -> "requir"."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def stems(text: str) -> set[str]:
    return {stem(w) for w in content_words(text) if w not in _GENERIC}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def check_question(question: str) -> str | None:
    q = question.strip()
    if not 15 <= len(q) <= 300:
        return "question_length"
    if not q.endswith("?"):
        return "question_not_a_question"
    if len(words(q)) < 4:
        return "question_too_short"
    return None


def check_answer(answer: str) -> str | None:
    if len(words(answer)) < 1:
        return "answer_empty"
    if "don't know" in answer.lower() or "not mentioned" in answer.lower():
        return "answer_is_abstention"
    return None


def check_no_leakage(question: str, evidence: str, n: int = 8) -> str | None:
    """Reject questions that copy a long word run from the evidence (answer leakage)."""
    q, e = words(question), words(evidence)
    shingles = {tuple(e[i : i + n]) for i in range(len(e) - n + 1)}
    if any(tuple(q[i : i + n]) in shingles for i in range(len(q) - n + 1)):
        return "question_copies_evidence"
    return None


def check_paraphrase(question: str, evidence: str, max_containment: float = 0.4) -> str | None:
    """At most 40% of the question's (stemmed, non-generic) content words may come from
    the evidence.

    Containment rather than Jaccard: a long quote makes Jaccard small even when the
    question copies nearly every word from it (seen in the pilot run).
    """
    q = stems(question)
    if q and len(q & stems(evidence)) / len(q) > max_containment:
        return "not_paraphrased"
    return None


def check_answer_supported(
    answer: str, question: str, evidence: str, min_fraction: float = 0.5
) -> str | None:
    """The answer's *new* content (words not already in the question) must mostly come
    from the evidence.

    Catches answers built from outside the evidence. In the pilot, a "fact" about Pod
    grace periods was attached to evidence about CoreDNS caching, because both contained
    "30 seconds".
    """
    new = stems(answer) - stems(question)
    if new and len(new & stems(evidence)) / len(new) < min_fraction:
        return "answer_not_supported_by_evidence"
    return None


_CONTEXT_DEPENDENT = re.compile(
    r"\bpassages?\b|\b(the|this|that) (example|output|snippet|status message|above|"
    r"following|text|document|section|command output|manifest)\b|\bin this case\b|"
    r"\baccording to the\b",
    re.IGNORECASE,
)


def check_self_contained(question: str) -> str | None:
    """Reject questions that only make sense next to the source passage."""
    return "not_self_contained" if _CONTEXT_DEPENDENT.search(question) else None


_LINK_ONLY = re.compile(r"^\W*(see|refer to|read|learn more about)\b.*\]\(", re.IGNORECASE)


def check_substantive_quote(quote: str) -> str | None:
    """Reject evidence that is only a cross-reference ("See [X](...) for more")."""
    return "evidence_is_only_a_link" if _LINK_ONLY.search(quote) else None


def check_uses_both(answer: str, quote_a: str, quote_b: str, min_words: int = 2) -> str | None:
    """Multi-source answers must use words specific to *each* quote.

    Rejects pseudo multi-hop items that are answerable from one passage alone.
    """
    ans, a, b = stems(answer), stems(quote_a), stems(quote_b)
    if len(ans & (a - b)) < min_words or len(ans & (b - a)) < min_words:
        return "answer_does_not_use_both_sources"
    return None


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def perturb_number(statement: str, evidence: str) -> tuple[str, str, str] | None:
    """Replace the first number shared by statement and evidence with a different value.

    Returns (false_statement, original, replacement), or None if nothing to perturb.
    Deterministic: the replacement doubles the value (or adds 2 for 0 and 1), and must
    not occur anywhere in the evidence.
    """
    evidence_numbers = set(_NUMBER.findall(evidence))
    for m in _NUMBER.finditer(statement):
        original = m.group()
        if original not in evidence_numbers:
            continue
        value = float(original)
        new = value + 2 if value in (0, 1) else value * 2
        replacement = str(int(new)) if new.is_integer() and "." not in original else f"{new:g}"
        if replacement in evidence_numbers:
            continue
        false = statement[: m.start()] + replacement + statement[m.end() :]
        return false, original, replacement
    return None


def check_contains_term(question: str, term: str) -> str | None:
    return None if term.lower() in question.lower() else "missing_required_term"


def check_numeric_answer(answer: str, evidence: str) -> str | None:
    numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
    if not numbers:
        return "answer_has_no_number"
    if not numbers & set(re.findall(r"\d+(?:\.\d+)?", evidence)):
        return "number_not_in_evidence"
    return None


def check_subject_removed(ambiguous: str, specific_terms: Iterable[str]) -> str | None:
    lowered = ambiguous.lower()
    if any(t.lower() in lowered for t in specific_terms if t):
        return "ambiguous_question_names_subject"
    return None


class DuplicateGuard:
    """Rejects exact and near-duplicate questions (content-word Jaccard >= threshold)."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.seen: list[set[str]] = []

    def check(self, question: str) -> str | None:
        cw = content_words(question)
        if any(jaccard(cw, s) >= self.threshold for s in self.seen):
            return "near_duplicate"
        return None

    def add(self, question: str) -> None:
        self.seen.append(content_words(question))
