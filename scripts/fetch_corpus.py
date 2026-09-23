"""Fetch the pinned Kubernetes docs corpus and write its manifest.

Usage:
    python scripts/fetch_corpus.py [--dest data/raw/kubernetes-website]

The checkout is gitignored. The manifest (data/corpus/kubernetes/manifest.json) is
committed, so any drift in the corpus is visible in review.
"""

import argparse
import logging
from pathlib import Path

from app.ingestion.kubernetes import PINNED_COMMIT, KubernetesCorpus, fetch, write_manifest
from app.utils.logging import configure_logging

DEFAULT_DEST = Path("data/raw/kubernetes-website")
MANIFEST = Path("data/corpus/kubernetes/manifest.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--commit", default=PINNED_COMMIT)
    args = parser.parse_args()
    configure_logging()

    corpus = KubernetesCorpus(fetch(args.dest, args.commit))
    manifest = write_manifest(corpus, MANIFEST)
    logging.info(
        "Corpus ready: %d files, docs v%s, commit %s -> %s",
        manifest["file_count"],
        manifest["docs_version"],
        manifest["commit"][:12],
        MANIFEST,
    )
