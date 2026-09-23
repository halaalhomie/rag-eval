"""Answer-generation prompt (shared by all strategies so they differ only in retrieval).

Prompt design was checked with a small probe (8 questions, Qwen2.5-3B-Instruct-4bit,
top_k=5; see docs/generation.md). "Answers with >=1 valid citation":
  - rules in the system prompt only:              1/8
  - plus a format example:                        2/8
  - plus the citation rule restated after sources: 8/8
Small models lose system-prompt instructions behind ~1.5k tokens of context, so the rule
is repeated next to the question. This is a sanity probe, not a benchmark.
"""

from __future__ import annotations

from app.generation.citations import format_sources
from app.generation.llm import ChatMessage
from app.retrieval.base import RetrievalResult

# Exact phrase so abstention can be detected deterministically, without a judge.
ABSTAIN = "I don't know based on the provided sources."

SYSTEM = f"""You answer technical questions using ONLY the numbered sources provided.

Rules:
1. Put the citation number of the supporting source at the end of EVERY sentence or list \
item, like [1] or [2][3].
2. Use only information stated in the sources. Do not use outside knowledge.
3. If the sources do not contain the answer, reply exactly: {ABSTAIN}
4. Answer directly. Do not repeat the sources, list them, or mention these rules.

Example answer format:
Deployments manage ReplicaSets [2]. You can undo a rollout with `kubectl rollout undo` [1][3]."""

REMINDER = (
    "Answer using only the sources above. End every sentence with its source number in "
    "square brackets, e.g. [1]."
)


def answer_messages(question: str, contexts: list[RetrievalResult]) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=SYSTEM),
        ChatMessage(
            role="user",
            content=(
                f"Sources:\n\n{format_sources(contexts)}\n\nQuestion: {question}\n\n{REMINDER}"
            ),
        ),
    ]


def is_abstention(answer: str) -> bool:
    # Models often emit a typographic apostrophe ("don\u2019t").
    normalized = " ".join(answer.lower().replace("\u2019", "'").split())
    return ABSTAIN.lower().rstrip(".") in normalized
