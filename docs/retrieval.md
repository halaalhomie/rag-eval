# Retrieval

This document covers the retriever interface, and each retrieval method as it is
implemented. Current status: **dense retrieval**, **BM25** and **hybrid fusion**. Reranking
and parent-child expansion are added in later phases.

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

## BM25

[app/retrieval/bm25.py](../app/retrieval/bm25.py): Okapi BM25 over an in-process inverted
index.

```
score(q, d) = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·|d|/avgdl))
idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))        # Lucene form, never negative
```

- **Real BM25, not Postgres full-text search.** `ts_rank` has no IDF saturation and a
  different length normalization, so a "BM25" row built on it would be mislabelled.
- **In memory, rebuilt automatically.** The index (3,855 chunks, 21,030 terms) builds in
  about 1.1 s including the database load. Before each query, a cheap
  `count(*), max(created_at)` check detects re-ingestion and triggers a rebuild, so the
  index can't go stale.
- **Same indexed text as dense retrieval.** Each chunk is indexed with its
  `Title > Section` prefix (`BM25_INCLUDE_CONTEXT`), so the two retrievers see the same
  input.
- **Deterministic and filterable.** Ties break by chunk ID. Metadata filters are applied
  after scoring, with a deeper candidate list.
- **Latency:** 3.9 ms p50 and 5.5 ms p95 per query, versus 24 ms for dense retrieval,
  because no query embedding is computed.

### Tokenization for technical text

The [analyzer](../app/retrieval/text.py) always lowercases and drops stopwords. On top of
that it can emit:

- **compounds:** whole identifiers such as `kube-apiserver`, `spec.replicas`,
  `grace-period`, alongside their parts
- **CamelCase parts:** `PodDisruptionBudget` also yields `pod`, `disruption`, `budget`
- **light stems:** the same rules as the dataset validators

### Choosing the configuration on the dev split

Every choice below was made on the **dev split** (88 answerable items). The test split was
evaluated once, with the final configuration.

| Analyzer (k1 = 1.2, b = 0.75) | EvR@5 | EvR@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| *dense retrieval (reference)* | *0.748* | *0.847* | *0.617* | *0.594* |
| words | 0.818 | 0.871 | 0.780 | 0.729 |
| words + stem | 0.820 | 0.849 | 0.782 | 0.731 |
| **words + compound** (chosen) | **0.827** | **0.871** | **0.788** | **0.737** |
| words + compound + camel | 0.817 | 0.869 | 0.752 | 0.712 |
| words + compound + camel + stem | 0.820 | 0.850 | 0.777 | 0.726 |

**No analyzer variant is significantly better than plain words.** Every paired-bootstrap
95% interval of the difference includes 0. Compounds changed only 6 of 88 queries (4
better, 2 worse). CamelCase splitting leaned negative on MRR (12 queries worse, 4 better;
difference −0.027, CI [−0.061, +0.005]): splitting identifiers into common words such as
`pod` dilutes their specificity. We chose words + compound because it had the best point
estimate on every metric and only affects queries that contain identifiers. Stemming and
CamelCase parts stay available as switches.

**k1/b grid (words + compound, dev):** the grid is flat. EvR@5 ranges 0.827–0.845 and MRR
0.775–0.793 over k1 ∈ {0.9, 1.2, 1.5} × b ∈ {0.5, 0.75, 0.9}. That spread is one or two
queries out of 88. Picking the best cell would fit noise, so the standard defaults
(k1 = 1.2, b = 0.75) are kept. All ablation reports are in
[data/experiments/ablations/](../data/experiments/ablations/).

### Test-split results

See [evaluation.md](evaluation.md#bm25-vs-dense-test-split). BM25 is significantly better
than dense retrieval overall (Evidence Recall@5 0.791 vs 0.701; paired difference +0.090,
95% CI [+0.034, +0.143]). It is worse on comparison questions, and neither retriever helps
ambiguous questions. **Part of BM25's lead comes from the lexical bias of the synthetic
questions**, which is discussed there.

## Hybrid retrieval

[app/retrieval/hybrid.py](../app/retrieval/hybrid.py). Dense retrieval and BM25 each
return `RETRIEVAL_CANDIDATES` (50) results, and the two lists are fused:

| `FUSION_METHOD` | Fused score | Notes |
|---|---|---|
| `rrf` | Σ<sub>r</sub> w<sub>r</sub> / (`RRF_K` + rank<sub>r</sub>) | Weighted Reciprocal Rank Fusion. Rank-only, so the incompatible score scales (cosine vs unbounded BM25) never interact |
| `linear` (default) | Σ<sub>r</sub> w<sub>r</sub> · minmax<sub>r</sub>(score) | Each list is min-max normalized over its own candidates; sensitive to score distributions |

Weights come from `DENSE_WEIGHT` and `BM25_WEIGHT` and are never hard-coded. A chunk found
by only one retriever gets nothing from the other. Every fused result carries
`components`, its rank and score in each list, and `/query` returns them. So it is always
visible why a chunk ranked where it did. Latency: 39 ms p50 (both retrievers at depth 50,
plus fusion).

### How complementary are the two retrievers? (dev)

On dev, BM25's and dense retrieval's top 5 together cover **0.911** of the required
evidence spans, against 0.827 for BM25 and 0.748 for dense alone. 15 spans are found only
by dense, 30 only by BM25, and 25 by neither. That is the headroom fusion can exploit,
although a fused top 5 cannot hold all ten candidates.

### Choosing the fusion configuration (dev)

**Selection rule, fixed before tuning:** start from the textbook default (RRF, equal
weights, k = 60). Replace it only with a configuration that improves dev Evidence
Recall@5 by more than one query's worth (1/88 ≈ 0.011); break ties on MRR.

| Configuration (dense weight; BM25 = 1 − w) | EvR@5 | EvR@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| RRF, w = 0.5, k = 60 (default) | 0.826 | 0.917 | 0.780 | 0.730 |
| RRF, w = 0.3 / 0.4 / 0.6 / 0.7, k = 60 | 0.826 / 0.826 / 0.803 / 0.819 | 0.911–0.917 | 0.703–0.805 | |
| RRF, w = 0.3 / 0.4 / 0.5 / 0.6 / 0.7, k = 10 | 0.852 / 0.860 / 0.849 / 0.825 / 0.830 | 0.874–0.933 | 0.695–0.815 | |
| linear, w = 0.3 / 0.4 / 0.6 / 0.7 | 0.849 / 0.854 / 0.843 / 0.830 | 0.894–0.917 | 0.731–0.819 | |
| **linear, w = 0.5 (chosen)** | **0.866** | 0.917 | 0.800 | 0.756 |

Findings on dev:

- **The textbook default is no better than BM25 alone** (paired difference in EvR@5
  +0.001). Equal-weight RRF at k = 60 lets the weaker dense list dilute BM25.
- **RRF with k = 10 beats k = 60 at every weight.** A smaller k favours documents ranked
  near the top of either list. This pattern is consistent, not a single lucky cell.
- **Weighting dense above 0.5 hurts throughout**, consistent with BM25 being the stronger
  retriever on this data.
- **The winner, linear w = 0.5, improves EvR@5 by +0.040** (paired 95% CI [+0.011,
  +0.085]). The evidence is thin, though: only **4 of 88** queries changed, all for the
  better. Choosing the maximum of 15 configurations on 88 queries invites a winner's-curse
  effect, so the test result is the number to trust. RRF (preferably with k = 10) stays one
  setting away.

All 15 dev reports are in [data/experiments/ablations/](../data/experiments/ablations/).

### Test-split results

See [evaluation.md](evaluation.md#hybrid-vs-bm25-and-dense-test-split). Hybrid is
significantly better than dense retrieval, and **statistically indistinguishable from
BM25 overall**. It is better than BM25 on semantic, multi-hop and comparison questions,
and worse on numerical ones.

## Limitations

- `bge-small` was chosen to fit an 8 GB machine next to the LLM. Larger embedders (e.g.
  `bge-base`, 768-d) would likely retrieve better, but that needs a re-init and re-embed;
  it is a candidate experiment.
- There is no query-side caching: the same query is re-embedded on every call.
- HNSW is approximate. At about 4k vectors, exact search would be cheap, but HNSW keeps
  behaviour representative of larger corpora.
