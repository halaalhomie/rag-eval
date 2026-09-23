"""Kubernetes documentation corpus: pinned fetch, file discovery, URLs and a manifest.

Source: https://github.com/kubernetes/website (content/en/docs), licensed CC BY 4.0.
The corpus is pinned to a single commit so that the evaluation dataset, which references
document text, is reproducible. The manifest records a sha256 for every file, so drift
is detectable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.ingestion.hugo import HugoShortcodeResolver, load_glossary
from app.ingestion.markdown import split_front_matter
from app.ingestion.types import LoadedDocument

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/kubernetes/website.git"
PINNED_COMMIT = "5dd61e17b222c0220281657b843a3e15c86b3cfe"  # 2026-09-23, docs v1.37
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
DOCS_ROOT = "content/en/docs"
SUBPATHS = ("concepts", "tasks", "reference/glossary")
GLOSSARY_SUBPATH = "reference/glossary"
SOURCE_PREFIX = "kubernetes/"
SITE = "https://kubernetes.io/docs/"

_LATEST = re.compile(r'^\s*latest\s*=\s*"(v?\d+\.\d+)"', re.MULTILINE)


@dataclass(frozen=True, slots=True)
class KubernetesCorpus:
    checkout: Path  # local clone of kubernetes/website

    @property
    def docs_dir(self) -> Path:
        return self.checkout / DOCS_ROOT

    def docs_version(self) -> str:
        m = _LATEST.search((self.checkout / "hugo.toml").read_text(encoding="utf-8"))
        if not m:
            raise RuntimeError("could not find 'latest' version in hugo.toml")
        return m.group(1).lstrip("v")

    def files(self) -> list[tuple[Path, str]]:
        """(path, source) pairs in a deterministic order.

        `source` is the stable identifier the document ID is derived from.
        """
        out = []
        for sub in SUBPATHS:
            for path in sorted((self.docs_dir / sub).rglob("*.md")):
                rel = path.relative_to(self.docs_dir).as_posix()
                if rel == f"{GLOSSARY_SUBPATH}/_index.md":
                    continue  # rendered glossary listing page, no prose of its own
                out.append((path, SOURCE_PREFIX + rel))
        return out

    def resolver(self) -> HugoShortcodeResolver:
        return HugoShortcodeResolver(
            docs_version=self.docs_version(),
            glossary=load_glossary(self.docs_dir / GLOSSARY_SUBPATH),
        )


def url_for(source: str, front_matter_id: str | None = None) -> str:
    """Canonical kubernetes.io URL for a corpus source, used in citations."""
    rel = source.removeprefix(SOURCE_PREFIX).removesuffix(".md")
    if rel.startswith(GLOSSARY_SUBPATH + "/"):
        term = front_matter_id or rel.rsplit("/", 1)[-1]
        return f"{SITE}reference/glossary/?all=true#term-{term}"
    rel = rel.removesuffix("_index").rstrip("/")
    return f"{SITE}{rel}/" if rel else SITE


def kubernetes_metadata(path: Path, doc: LoadedDocument) -> dict[str, Any]:
    front_matter, _ = split_front_matter(path.read_text(encoding="utf-8"))
    rel = doc.source.removeprefix(SOURCE_PREFIX)
    return {
        "corpus": "kubernetes",
        "corpus_commit": PINNED_COMMIT,
        "url": url_for(doc.source, front_matter.get("id")),
        "doc_category": rel.split("/", 1)[0],  # concepts | tasks | reference
        "license": LICENSE,
    }


def fetch(dest: Path, commit: str = PINNED_COMMIT) -> Path:
    """Shallow, sparse checkout of the pinned commit (only the docs subpaths we use)."""
    dest.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(dest), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if not (dest / ".git").exists():
        git("init", "-q")
        git("remote", "add", "origin", REPO_URL)
    git("sparse-checkout", "set", *[f"{DOCS_ROOT}/{s}" for s in SUBPATHS])
    if _head(dest) != commit:
        logger.info("Fetching kubernetes/website@%s", commit[:12])
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
        git("checkout", "-q", "--detach", commit)
    return dest


def _head(dest: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def build_manifest(corpus: KubernetesCorpus) -> dict[str, Any]:
    files = [
        {
            "source": source,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path, source in corpus.files()
    ]
    return {
        "corpus": "kubernetes",
        "repository": REPO_URL,
        "commit": _head(corpus.checkout),
        "docs_version": corpus.docs_version(),
        "license": LICENSE,
        "attribution": "Kubernetes Documentation, The Kubernetes Authors",
        "subpaths": [f"{DOCS_ROOT}/{s}" for s in SUBPATHS],
        "file_count": len(files),
        "files": files,
    }


def write_manifest(corpus: KubernetesCorpus, path: Path) -> dict[str, Any]:
    manifest = build_manifest(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
