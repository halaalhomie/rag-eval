# RAG-Forge

An evaluation-driven adaptive RAG system. RAG-Forge decides per query how much retrieval
and verification is needed, and every technique it uses (BM25, hybrid fusion,
cross-encoder reranking, parent-child retrieval, query rewriting, corrective retrieval, a
Self-RAG-inspired loop, claim verification) can be switched on or off and benchmarked
against a baseline.

> **Status: under active development.** This README grows phase by phase, and it only
> describes what exists in the repository. Benchmark numbers appear only after the
> evaluation pipeline has actually produced them.

## Why this project exists

Most RAG demos run `query → vector search → LLM` and judge quality by eye. RAG-Forge is
built to answer measurable questions instead:

- Does hybrid retrieval beat dense-only on this corpus, and on which query types?
- How much does a cross-encoder reranker add to Recall@5 and MRR, and at what latency cost?
- When does corrective retrieval fire, and does it improve faithfulness enough to justify
  the extra LLM calls?
- What does each quality gain cost in latency, LLM calls and dollars?

## Target pipeline

```mermaid
flowchart LR
    Q[Query] --> A[Query analysis]
    A --> S{Strategy selection}
    S --> R[Retrieval<br/>dense / BM25 / hybrid]
    R --> RR[Cross-encoder rerank]
    RR --> G{Relevance grading}
    G -- insufficient --> CR[Rewrite + corrective retrieval]
    CR --> RR
    G -- sufficient --> E{Evidence sufficiency}
    E --> GEN[Generation with citations]
    GEN --> V{Claim verification}
    V -- unsupported --> GEN
    V -- supported --> F[Final answer + trace]
```

Every loop has a configurable maximum iteration count, and those counts are capped in
config validation.

## Implementation progress

| Phase | Component | Status |
|---|---|---|
| 1 | Repository, configuration, Docker, DB schema, `/health` | done |
| 2 | Document ingestion (MD/TXT/PDF), structure-aware parent/child chunking, Kubernetes corpus | done |
| 3 | Baseline dense RAG | planned |
| 4 | Evaluation dataset + retrieval metrics | planned |
| 5–8 | BM25, hybrid (RRF), reranking, parent-child | planned |
| 9–12 | Query rewriting, Corrective RAG, Self-RAG-inspired loop, claim verification | planned |
| 13–18 | Observability, experiment runner, benchmarks, docs, deployment | planned |

## Corpus

The evaluation corpus is the **Kubernetes documentation** (`concepts/`, `tasks/` and the
glossary from [kubernetes/website](https://github.com/kubernetes/website)). It is pinned
to commit `5dd61e1` (docs v1.37) and licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), © The Kubernetes Authors.
It suits RAG evaluation because it includes:

- factual lookups (defaults, field names)
- keyword-heavy queries (flags, feature gates)
- troubleshooting
- multi-hop questions (e.g. Deployment → ReplicaSet → Pod)
- comparisons (StatefulSet vs Deployment)

Current ingest: 551 documents, 947 parent chunks and 3,062 child chunks. See
[docs/ingestion.md](docs/ingestion.md) for chunking design, measurements and limitations.

## Benchmark results

No experiments have been run yet. All values stay **TBD** until they are produced by
`scripts/run_experiments.py` and recorded under `data/experiments/`.

| System | Recall@5 | MRR | Faithfulness | Answer Relevance | Avg Latency | Cost / query |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | TBD | TBD | TBD | TBD | TBD | TBD |
| BM25 | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid + Reranker | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid + Reranker + Rewrite | TBD | TBD | TBD | TBD | TBD | TBD |
| Corrective RAG | TBD | TBD | TBD | TBD | TBD | TBD |
| Self-RAG-inspired | TBD | TBD | TBD | TBD | TBD | TBD |

## Setup

Prerequisites: Docker, and Python 3.12+ for local development.

```bash
cp .env.example .env            # then set ANTHROPIC_API_KEY (or another provider)

# Full stack: PostgreSQL + pgvector and the API (schema is initialized on startup)
docker compose up -d --build
curl localhost:8000/health
```

Fetch and ingest the corpus:

```bash
python scripts/fetch_corpus.py                 # pinned sparse checkout into data/raw/
python scripts/ingest.py --corpus kubernetes   # idempotent; re-runs skip unchanged docs
python scripts/ingest.py --path ./my-docs      # or any folder of .md / .txt / .pdf
```

Local development against the containerized database:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d postgres
python scripts/init_db.py
uvicorn app.main:app --reload
```

Postgres is published on host port **5434** by default, to avoid clashing with a local
Postgres on 5432. Override it with `POSTGRES_HOST_PORT`, and keep `DATABASE_URL` in sync.

### Tests

```bash
pytest                    # unit + integration (integration tests skip if Postgres is down)
pytest -m "not integration"
```

Integration tests create and use a separate `ragforge_test` database.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | database and pgvector status |
| `POST` | `/documents/ingest` | multipart upload of `.md` / `.txt` / `.pdf` files (≤20 files, ≤20 MB each), with per-file status |
| `GET` | `/documents` | list ingested documents with chunk counts and source URLs |

```bash
curl -X POST localhost:8000/documents/ingest -F "files=@runbook.md" -F "files=@manual.pdf"
```

Each file succeeds or fails independently: unsupported types, corrupt PDFs and empty files
are reported per file instead of failing the whole request.

## Configuration

All tunables live in [app/config/settings.py](app/config/settings.py) and are validated at
startup. [.env.example](.env.example) documents every variable. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_STRATEGY` | `adaptive` | `baseline`, `dense`, `bm25`, `hybrid`, `hybrid_rerank`, `hybrid_rerank_rewrite`, `corrective`, `self_rag`, `adaptive` |
| `TOP_K` / `RERANK_TOP_K` | 10 / 5 | final context counts |
| `CHILD_CHUNK_SIZE` / `CHILD_CHUNK_OVERLAP` / `PARENT_CHUNK_SIZE` | 400 / 60 / 1600 | chunking (approximate tokens) |
| `DENSE_WEIGHT` / `BM25_WEIGHT` / `RRF_K` | 0.5 / 0.5 / 60 | fusion |
| `RELEVANCE_THRESHOLD` / `EVIDENCE_THRESHOLD` | 0.65 / 0.70 | corrective and sufficiency gates |
| `MAX_CORRECTIVE_ITERATIONS` / `MAX_SELF_RAG_ITERATIONS` | 2 / 2 | loop caps (validated ≤ 5) |
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai_compatible`, `fake` |
| `ENABLE_LANGFUSE` | `false` | tracing (keys are required when enabled) |

Cross-field rules are enforced. For example, `RERANK_TOP_K ≤ RERANK_CANDIDATES ≤
RETRIEVAL_CANDIDATES`, and the fusion weights cannot both be zero.

## Project structure

```
app/
  api/            FastAPI routes
  config/         validated settings
  db/             SQLAlchemy models, session, schema init
  ingestion/      loaders, Hugo preprocessing, parent/child chunking, pipeline
  retrieval/      dense, BM25, hybrid, reranker      (phases 3–8)
  rag/            LangGraph pipeline, nodes, graders (phases 9–12)
  generation/     LLM providers, answer generation   (phase 3)
  evaluation/     datasets, metrics, experiments     (phase 4+)
  observability/  tracing                            (phase 13)
scripts/          CLI entry points (init_db, ingest, eval, experiments)
tests/            unit / integration / evaluation
docs/             technical documentation
data/             corpus manifest, raw corpus (gitignored), eval sets, experiment outputs
```

## Documentation

- [docs/architecture.md](docs/architecture.md): system design, data model, key decisions
- [docs/ingestion.md](docs/ingestion.md): loaders, chunking algorithm, corpus, measurements
