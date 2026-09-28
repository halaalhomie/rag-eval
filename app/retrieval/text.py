"""Tokenization for lexical retrieval over technical documentation.

Plain word splitting treats `kube-apiserver`, `spec.replicas`, `--grace-period` and
`PodDisruptionBudget` badly: the identifier a user typed exactly is lost among its common
parts. The analyzer can emit, per the options:

- words: lowercase alphanumeric runs ("kube", "apiserver")
- compounds: whole identifiers joined by - . _ / ("kube-apiserver", "spec.replicas")
- camel parts: CamelCase components ("pod", "disruption", "budget") next to the whole word
- stems: light suffix stripping ("probes" -> "probe")

Stopwords are dropped (compounds are never stopwords). Which options help is an empirical
question, answered on the dev split (see docs/retrieval.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
_COMPOUND = re.compile(r"[A-Za-z0-9]+(?:[-._/][A-Za-z0-9]+)+")
_ALNUM = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")

STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have how i if in "
    "into is it its may might must no not of on or our should so such than that the their "
    "them then there these they this those to was we were what when where which while who "
    "why will with would you your".split()
)


def stem(word: str) -> str:
    """Light suffix stripping (same rules as the dataset validators)."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


@dataclass(frozen=True)
class Analyzer:
    compounds: bool = True
    camel_parts: bool = True
    stemming: bool = True

    def name(self) -> str:
        flags = [
            f
            for f, on in (
                ("compound", self.compounds),
                ("camel", self.camel_parts),
                ("stem", self.stemming),
            )
            if on
        ]
        return "+".join(["words", *flags])

    def __call__(self, text: str) -> list[str]:
        tokens: list[str] = []
        for w in _WORD.findall(text.lower()):
            if w not in STOPWORDS:
                tokens.append(stem(w) if self.stemming else w)
        if self.compounds:
            tokens += [c.lower() for c in _COMPOUND.findall(text)]
        if self.camel_parts:
            for run in _ALNUM.findall(text):
                parts = _CAMEL.findall(run)
                if len(parts) > 1 and any(p[0].isupper() for p in parts[1:]):
                    tokens += [stem(p.lower()) if self.stemming else p.lower() for p in parts]
        return tokens
