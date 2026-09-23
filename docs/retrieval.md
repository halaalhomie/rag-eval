# Retrieval

This document covers the retriever interface, and each retrieval method as it is
implemented. Current status: **dense retrieval**. BM25, hybrid fusion, reranking and
parent-child expansion are added in later phases.

## Interface

Every retriever implements one method and returns one result type
([app/retrieval/base.py](../app/retrieval/base.py)):

```python
retrieve(query: str, k: int, filters: dict | None = None) -> list[RetrievalResult]
```

| Field | Meaning |
|---|---|
| `document_id`, `chunk_id`, `parent_id` | stable identifiers; the chunk is the unit that was scored |
| `text` | chunk text, an exact slice `documents.content[start_char:end_char]` |
| `score` | retriever-specific; **higher is better** for every retriever |
| `rank` | 1-based position in this retriever's output |
| `retriever` | which retriever produced it (`dense`, and later `bm25`, `hybrid`, ...) |
| `section`, `page`, `metadata` | heading path, PDF page, and `title` / `url` / `doc_category` |

Scores from different retrievers are **not comparable**. A cosine similarity and a BM25
score live on different scales, which is why hybrid retrieval will fuse by rank (RRF)
rather than by raw score.

Pipelines and the evaluation framework depend only on this interface. Swapping dense for
BM25 or hybrid is a configuration change, which is what makes the retrieval experiments
comparable.

## Dense retrieval

[app/retrieval/dense.py](../app/retrieval/dense.py): PostgreSQL + pgvector (0.8.6), HNSW
index.

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-d, 33M parameters, 512-token input limit; runs locally |
| `EMBEDDING_QUERY_PREFIX` | BGE retrieval instruction | added to queries only (asymmetric retrieval) |
| `EMBEDDING_INCLUDE_CONTEXT` | `true` | chunks are embedded as `Title > Section` + text |
| `SIMILARITY_METRIC` | `cosine` | `cosine`, `l2` or `inner_product`; the HNSW operator class must match (set at `init_db`) |
| `EMBEDDING_DEVICE` | `cpu` | `mps` on Apple Silicon embedded about 1.8× faster per chunk in bulk ingestion (43 vs 79 ms) |

Embeddings are L2-normalized, so cosine, inner product and L2 give the same ranking. The
integration tests check that all three agree. The setting exists so that
non-normalized models can be used.

Scores are converted from distances so that higher is better: cosine gives `1 − distance`
(similarity), L2 gives `−distance`, and inner product gives the inner product itself.

### Details that silently change results

These are easy to miss, and each one is handled explicitly:

1. **`hnsw.ef_search` caps result count.** pgvector's default is 40, so asking for 50
   candidates would quietly return 40. That would bias every "top-50 then rerank"
   experiment. The retriever sets `ef_search = max(40, 2k)` per query. Verified on the
   corpus: `k=60` returns 60 results.
2. **Filtered HNSW search can under-return.** HNSW finds neighbours first and filters
   second, so a selective filter can leave fewer than `k` rows. With a filter, the
   retriever enables pgvector 0.8's `iterative_scan = relaxed_order` and re-sorts the rows.
3. **Mixed embedding models.** Rows are restricted to
   `embedding_model = <current model>`. After a model switch, stale vectors can never be
   ranked against new query vectors.
4. **Embedder truncation.** Handled at ingestion (see
   [ingestion.md](ingestion.md#token-estimation)): 0 of 3,855 embedding inputs exceed
   bge-small's 512-token limit.

### Metadata filtering

`filters` is JSON containment (`@>`) on chunk metadata, backed by a GIN index. For example,
`{"doc_category": "tasks"}` restricts retrieval to how-to pages. Document-level fields
(`title`, `url`, `doc_category`, `corpus`) are copied onto each chunk at ingestion, so
filtering needs no join.

### Observed behaviour (not a benchmark)

Spot checks on the ingested corpus, in the local dev environment (M1, 8 GB RAM). Query
latency is about **25 ms** end to end, including query embedding, once the model is
loaded. The first query after process start pays about 10 s of model loading, so the API
preloads the embedder at startup.

| Query | Top result (section path) |
|---|---|
| What is the default termination grace period for a Pod? | Pod Lifecycle > Termination of Pods > Pod Termination Flow |
| How do I roll back a Deployment to a previous revision? | Deployments > Rolling Back a Deployment |
| how to debug a pod stuck in CrashLoopBackOff (filter: tasks) | Debug Running Pods > Debugging using a copy of the Pod |

Retrieval quality is measured with Recall@K, MRR and NDCG in the evaluation phase. These
examples only confirm that the pipeline is wired correctly.

## Limitations

- `bge-small` was chosen to fit an 8 GB machine next to the LLM. Larger embedders (e.g.
  `bge-base`, 768-d) would likely retrieve better, but that needs a re-init and re-embed;
  it is a candidate experiment.
- There is no query-side caching: the same query is re-embedded on every call.
- HNSW is approximate. At about 4k vectors, exact search would be cheap, but HNSW keeps
  behaviour representative of larger corpora.
