"""Resolve the Hugo shortcodes used by the kubernetes/website docs into plain text.

Without this step BM25 and the embedder would index template syntax such as
`{{< glossary_tooltip term_id="node" >}}` instead of the word "node". The goal is to
approximate what a reader sees on kubernetes.io, not to reimplement Hugo:

- glossary tooltips/definitions are resolved from the glossary source files
- version shortcodes are substituted with the pinned docs version
- admonitions (note/caution/warning) become an inline "Note:" label
- boilerplate includes and version checks are dropped
- code samples referencing files outside the docs tree become a short placeholder
- unknown shortcodes are stripped (their inner content is kept)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.markdown import split_front_matter

logger = logging.getLogger(__name__)

_SHORTCODE = re.compile(r"\{\{([<%])\s*(/?)\s*([\w-]+)(.*?)\s*[>%]\}\}", re.DOTALL)
_ARG = re.compile(r'([\w-]+)="([^"]*)"|"([^"]*)"|(\S+)')
# Blocks whose inner content is not reader-visible prose.
_DROP_BLOCKS = re.compile(
    r"\{\{[<%]\s*(comment|mermaid)\b.*?[>%]\}\}.*?\{\{[<%]\s*/\s*\1\s*[>%]\}\}", re.DOTALL
)
_MORE = "<!--more-->"
_MAX_DEPTH = 3  # glossary definitions can contain shortcodes; bound the recursion

_HEADINGS = {
    "whatsnext": "What's next",
    "prerequisites": "Before you begin",
    "cleanup": "Clean up",
    "objectives": "Objectives",
    "synopsis": "Synopsis",
    "options": "Options",
    "seealso": "See also",
}
_ADMONITIONS = {"note": "Note:", "caution": "Caution:", "warning": "Warning:"}
_DROP = {"include", "version-check", "thirdparty-content", "toc", "legacy-repos-deprecation"}
# Pure wrappers: the tag disappears, the content between open/close tags is kept.
_WRAPPERS = {"tabs", "highlight", "table", "example", "pageinfo"}


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    term_id: str
    title: str
    short: str
    full: str


def load_glossary(glossary_dir: Path) -> dict[str, GlossaryEntry]:
    entries: dict[str, GlossaryEntry] = {}
    for path in sorted(glossary_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        term_id = str(meta.get("id") or path.stem)
        short, _, rest = body.partition(_MORE)
        entries[term_id] = GlossaryEntry(
            term_id=term_id,
            title=str(meta.get("title") or term_id),
            short=short.strip(),
            full=(short.strip() + "\n\n" + rest.strip()).strip(),
        )
    return entries


def _parse_args(raw: str) -> tuple[dict[str, str], list[str]]:
    named: dict[str, str] = {}
    positional: list[str] = []
    for m in _ARG.finditer(raw):
        if m.group(1):
            named[m.group(1)] = m.group(2)
        else:
            positional.append(m.group(3) if m.group(3) is not None else m.group(4))
    return named, positional


class HugoShortcodeResolver:
    """Callable preprocessor: `resolver(markdown_body) -> markdown_body`."""

    def __init__(self, docs_version: str, glossary: dict[str, GlossaryEntry] | None = None):
        major, minor = docs_version.lstrip("v").split(".")[:2]
        self.major, self.minor = int(major), int(minor)
        self.glossary = glossary or {}
        self.unknown: set[str] = set()

    def __call__(self, text: str) -> str:
        return self._resolve(text, depth=0)

    def _resolve(self, text: str, depth: int) -> str:
        text = _DROP_BLOCKS.sub("", text)
        return _SHORTCODE.sub(lambda m: self._replace(m, depth), text)

    def _version(self, minor_offset: int = 0) -> str:
        return f"{self.major}.{self.minor + minor_offset}"

    def _replace(self, m: re.Match[str], depth: int) -> str:
        closing, name, raw_args = m.group(2) == "/", m.group(3), m.group(4)
        if closing:
            return ""
        named, pos = _parse_args(raw_args)
        match name:
            case "glossary_tooltip":
                term = self.glossary.get(named.get("term_id", ""))
                return named.get("text") or (term.title if term else named.get("term_id", ""))
            case "glossary_definition":
                term = self.glossary.get(named.get("term_id", ""))
                if term is None or depth >= _MAX_DEPTH:
                    return ""
                body = term.full if named.get("length") == "all" else term.short
                body = self._resolve(body, depth + 1)
                prepend = named.get("prepend", "")
                # Hugo lowercases the first letter of the definition after a prepend.
                if prepend and body:
                    body = body[0].lower() + body[1:]
                return f"{prepend} {body}".strip()
            case "heading":
                key = pos[0] if pos else ""
                return _HEADINGS.get(key, key.replace("-", " ").capitalize())
            case "skew":
                what = pos[0] if pos else "currentVersion"
                if what == "currentVersionAddMinor" and len(pos) >= 2:
                    return self._version(int(pos[1]))
                if what == "currentPatchVersion":
                    return f"{self._version()}.x"  # patch level is not pinned by the docs
                return self._version()
            case "param":
                return f"v{self._version()}" if pos and pos[0] == "version" else ""
            case "feature-state":
                if "for_k8s_version" in named:
                    state = named.get("state", "")
                    return f"FEATURE STATE: Kubernetes {named['for_k8s_version']} [{state}]"
                if "feature_gate_name" in named:
                    return f"FEATURE STATE: feature gate `{named['feature_gate_name']}`"
                return ""
            case "code_sample" | "codenew":
                file = named.get("file", "")
                return f"(Example manifest: `{file}`)" if file else ""
            case "tab":
                return f"{named['name']}:" if "name" in named else ""
            case "link":
                return named.get("text", "")
            case "relref":
                return pos[0] if pos else ""  # sits inside a Markdown link target
            case "api-reference":
                # page="core/persistent-volume-v1" renders as the kind: PersistentVolume
                kind = named.get("page", "").rsplit("/", 1)[-1]
                kind = re.sub(r"-v\d+(?:(?:alpha|beta)\d+)?$", "", kind)
                return "".join(part.capitalize() for part in kind.split("-"))
            case "alert":
                return f"{named['title']}:" if "title" in named else ""
            case "details":
                return named.get("summary", "")
            case "figure":
                caption = named.get("caption") or named.get("title") or named.get("alt")
                return f"Figure: {caption}" if caption else ""
            case _ if name in _ADMONITIONS:
                return _ADMONITIONS[name]
            case _ if name in _DROP or name in _WRAPPERS:
                return ""
            case _:
                self.unknown.add(name)
                return ""
