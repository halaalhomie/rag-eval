"""Ingest documents into the database.

Usage:
    python scripts/ingest.py --corpus kubernetes         # pinned K8s docs (fetch first)
    python scripts/ingest.py --path docs/ --path a.pdf   # arbitrary MD / TXT / PDF files

Re-running is idempotent: documents whose content and chunking config are unchanged are
skipped.
"""

import argparse
import json
import logging
from pathlib import Path

from app.config import get_settings
from app.db.init_db import init_db
from app.db.session import session_scope
from app.ingestion.kubernetes import KubernetesCorpus, kubernetes_metadata
from app.ingestion.loaders import SUFFIX_TYPES
from app.ingestion.pipeline import IngestionPipeline, corpus_stats
from app.retrieval.embeddings import get_embedder
from app.utils.logging import configure_logging

DEFAULT_CHECKOUT = Path("data/raw/kubernetes-website")


def _expand(paths: list[Path]) -> list[tuple[Path, str]]:
    files = []
    for p in paths:
        candidates = sorted(p.rglob("*")) if p.is_dir() else [p]
        files += [
            (f, f.resolve().as_posix())
            for f in candidates
            if f.is_file() and (not p.is_dir() or f.suffix.lower() in SUFFIX_TYPES)
        ]
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", choices=["kubernetes"])
    src.add_argument("--path", type=Path, action="append")
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--limit", type=int, help="ingest only the first N files")
    parser.add_argument("--no-embed", action="store_true", help="chunk only, skip embeddings")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.observability.log_level)
    init_db()
    embedder = None if args.no_embed else get_embedder(settings.embedding)
    pipeline = IngestionPipeline(settings, embedder)

    if args.corpus == "kubernetes":
        if not (args.checkout / "hugo.toml").exists():
            parser.error(f"{args.checkout} not found; run scripts/fetch_corpus.py first")
        corpus = KubernetesCorpus(args.checkout)
        files, pre, meta = corpus.files(), corpus.resolver(), kubernetes_metadata
    else:
        files, pre, meta = _expand(args.path), None, None
    files = files[: args.limit] if args.limit else files

    with session_scope() as session:
        summary = pipeline.ingest_files(session, files, preprocessor=pre, metadata_for=meta)
        stats = corpus_stats(session)
    for r in summary.results:
        if r.status == "failed":
            logging.error("FAILED %s: %s", r.source, r.error)
    if args.corpus == "kubernetes" and pre.unknown:
        logging.warning("Unhandled Hugo shortcodes (stripped): %s", sorted(pre.unknown))
    print(json.dumps({"run": summary.as_dict(), "database": stats}, indent=2))
