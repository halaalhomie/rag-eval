# Evaluation

How RAG-Forge is evaluated: the dataset, how relevance is labelled, the retrieval
metrics, and the baseline results. Generation metrics (faithfulness, answer relevance,
citation correctness) are added in a later phase.

> **The evaluation dataset is synthetic.** Questions, answers and evidence selections were
> produced by a local 3B LLM and filtered by deterministic validators. It is **not
> human-annotated**. See [Known weaknesses](#known-weaknesses-of-the-dataset).

## Dataset

`data/eval/kubernetes_v1.jsonl`, one item per line
([schema](../app/evaluation/datasets/schema.py)), with a companion `.meta.json` that
records the generator model, corpus commit, chunker fingerprint, seed, counts per type and
split, and the rejection statistics of every validator.

```json
{
  "id": "q_…",
  "question": "…",
  "ground_truth_answer": "…",
  "question_type": "multi_hop",
  "difficulty": "hard",
  "split": "test",
  "evidence": [{"document_id": "doc_…", "start_char": 1234, "end_char": 1391, "quote": "…"}],
  "relevant_document_ids": ["doc_…", "doc_…"],
  "relevant_chunk_ids": ["doc_…:c12", "doc_…:c3"],
  "subtype": "linked_documents",
  "notes": {"…": "…"},
  "provenance": {"method": "synthetic_llm", "generator_model": "…", "prompt_version": "qgen-v4",
                 "validators_passed": ["…"]}
}
```

### Relevance is anchored to evidence spans, not chunk IDs

`evidence` holds exact character spans in `documents.content`. `relevant_chunk_ids` is only
a snapshot under the chunking used at generation time. **Evaluation recomputes chunk
relevance from the spans against the current index**
([labels.py](../app/evaluation/datasets/labels.py)): a chunk is relevant to a span if it
covers at least 50% of the span's characters.

The same dataset can therefore evaluate different chunk sizes, overlaps or parent-child
settings. Chunking becomes an experiment variable instead of something that silently
invalidates the labels. A span that no current chunk covers is counted, reported as
`unmapped_evidence_spans`, and scored as a miss.

### Dev and test split

Items are split **30% dev / 70% test, stratified by question type**, with a fixed seed.
Thresholds (relevance, evidence and fusion weights) may be tuned only on **dev**; the
README benchmark table reports **test**. Tuning on the same questions that are reported
would overstate every tuned technique.

### Question types

| Type | How it is built | Validators beyond the common ones |
|---|---|---|
| `simple_factual` | one chunk → direct question | no leakage |
| `semantic` | one chunk → paraphrased question | ≤40% of the question's content words appear in the evidence |
| `keyword_heavy` | one chunk plus a required identifier (flag, dotted field, camelCase) | the identifier appears in both the question and the evidence |
| `numerical` | chunk containing number + unit | the answer's number appears in the evidence |
| `multi_hop` | chunk A **plus the chunk of a page A links to**, chosen by dense search within the linked page | the answer uses words specific to *each* source; neither evidence is only a link |
| `comparison` | intros of two **similar** concept pages in the same docs section | the answer uses both sources |
| `ambiguous` | specific question → vague rewrite ("What happens when it is deleted?") | the rewrite names neither the page title nor the subject; the specific form is kept in `notes` |
| `adversarial` | **code-built false premise**: the LLM states a true numeric fact, and code changes the number | the fact must be supported by its evidence; the true and false values are in `notes` |
| `unanswerable` / `nonexistent_feature` | LLM invents a plausible feature or flag | **the invented term appears nowhere in the corpus** (SQL check) |
| `unanswerable` / `out_of_domain` | question about another technology | **the topic keyword appears nowhere in the corpus**; no Kubernetes terms |

Common validators, applied to every item:
- the question is well formed and self-contained (no "the passage", "the example", "in this case")
- it is not a near-duplicate (content-word Jaccard below 0.8)
- the answer is non-empty and supported by the evidence (at least 50% of its new content words appear in it)
- the evidence maps to at least one indexed chunk

Sampling: each chunk is used at most once, there are at most 3 items per document, and
chunks that are mostly code are skipped.

`difficulty` is a **heuristic by type**: easy for simple, keyword and numerical; medium
for semantic, ambiguous, comparison and unanswerable; hard for multi-hop and adversarial.
It is not a measured property.

### How the generator was debugged (pilot runs)

Three pilot runs of 3 items per type were reviewed by reading every accepted item. Each
review changed the design:

| Pilot | Finding | Change |
|---|---|---|
| 1 | **3/3 adversarial items were wrong.** The 3B model wrote "false" premises that were true, or restated the passage | False premises are now built by code: a true numeric fact, with its number changed |
| 1 | Comparisons between unrelated "sibling" pages (metrics vs version emulation) | Pair each concept page with its *most similar* intro in the same section |
| 1 | One "multi-hop" item was answerable from passage A alone; its B evidence was "See [link] for more" | The answer must use both sources; link-only evidence is rejected |
| 1 | "Semantic" questions copied the passage, but Jaccard stayed low because the quotes were long | Containment instead of Jaccard |
| 1 | "…according to *the status message*?" | Self-containment check |
| 2 | **A fabricated fact**: "the default Pod grace period is 30 seconds", attached to evidence about CoreDNS caching for 30 seconds. The model had copied the example sentence from the prompt | Removed the example; added the answer-support check (the fact must come from its evidence) |
| 2 | **Multi-hop 0/3**: 14 of 18 attempts failed because the model could not copy two quotes verbatim | Passages shown as **numbered sentences**; the model returns sentence IDs and spans are mapped deterministically |
| 2 | "…differ between the two passages?" | "passage" is rejected in questions and answers |
| 2 | The paraphrase check was diluted by generic words ("Kubernetes cluster") | Light stemming, generic words excluded, threshold 0.4 |

### Dataset statistics (`kubernetes_v1`, version `c6009ef85e53`)

**316 items**: 219 test and 97 dev, covering 208 of the 551 documents. The set was
generated in two runs: seed 42 produced 275 items; seed 43 then topped up the short types
with 41 more, without reusing earlier chunks or questions. Generator:
`mlx-community/Qwen2.5-3B-Instruct-4bit`, prompt `qgen-v4` (the first run used `qgen-v4`
too; only the out-of-domain prompt gained an "angle" in the top-up).

| Type | Items | Accepted / attempts | Most common rejection |
|---|---:|---:|---|
| simple_factual | 60 | 60 / 100 (60%) | not self-contained (15) |
| semantic | 45 | 45 / 423 (11%) | not paraphrased (304) |
| keyword_heavy | 45 | 45 / 251 (18%) | required identifier missing from the question (150) |
| numerical | 30 | 30 / 59 (51%) | number not in evidence (12) |
| multi_hop | 37 | 37 / 432 (9%) | answer does not use both sources (262) |
| comparison | **19** (target 30) | 19 / 336 (6%) | answer does not use both sources (158) |
| ambiguous | 25 | 25 / 79 (32%) | rewrite still names the subject (25) |
| adversarial | 25 | 25 / 103 (24%) | stated fact had no number (40) |
| unanswerable | 30 | 30 / 118 (25%) | out-of-domain near-duplicates (81; fixed in the top-up) |
| **Total** | **316** | **316 / 1,901 (17%)** | |

Only 17% of generator outputs were kept. Comparison stayed below its target (19 of 30):
rather than loosen its checks, the shortfall is reported as-is.

### Sample audit (AI review, not human review)

After each run, a random sample was read in full by the AI assistant that built this
pipeline. This is a sanity check, **not** human annotation; `provenance.reviewed_by` is
left empty for every item. Findings on 33 sampled items:

- **Acceptable: 25 of 33.** This includes every sampled adversarial, keyword and semantic
  item. It also includes one unanswerable item whose label was wrong and has since been
  fixed (see the nonexistent-feature note below).
- **Flawed: 8 of 33.**
  - 3 multi-hop items whose answers are vague or only partly use the second source
  - 1 comparison with a fabricated contrast ("scheduler vs X.509 certificates")
  - 1 numerical item that took "8" from "8 hours late" and dropped the unit
  - 1 ambiguous item whose answer misses the question
  - 2 simple items with garbled question grammar

The sample is too small to estimate per-type error rates. `scripts/export_review_sheet.py`
exports a stratified sample for real human review. Only verdicts entered by a person
should be reported as human-validated accuracy.

### Known weaknesses of the dataset

- **Synthetic-question lexical bias.** Questions are written *from* the passage they are
  evaluated against, so they tend to share its vocabulary. This favours lexical retrieval
  (BM25) and is a known bias of synthetic RAG benchmarks. The `semantic` type (at most 40%
  word overlap) partly counters it.
- **Same model family as the system.** The generator (Qwen 3B) is also the default answer
  model. Retrieval metrics are unaffected; generation metrics must keep this in mind.
- **Answer quality varies by type.** Multi-hop and comparison items have correct evidence
  spans more often than they have good ground-truth answers. Their *retrieval* labels are
  more reliable than their *answer* labels.
- **Some questions are example-specific** ("How many replicas are initially scheduled for
  the Deployment 'nginx'?"): answerable from the corpus, but less natural than real queries.
- **Nonexistent-feature items sit next to real features.** For a fake `--force-delete`
  flag, the documented `--force` exists, so the ground truth accepts "this is not
  documented; the documented alternative is …" as correct, not only "I don't know".
- **Coverage.** 208 of 551 documents, at most 3 items per document; small per-type test
  counts (13–42).

## Retrieval metrics

Deterministic, binary relevance ([metrics/retrieval.py](../app/evaluation/metrics/retrieval.py)),
with standard definitions:

| Metric | Definition |
|---|---|
| Recall@K | \|relevant ∩ top-K\| / \|relevant\| (chunks) |
| Precision@K | \|relevant ∩ top-K\| / K |
| Hit@K | 1 if any relevant chunk is in the top-K |
| MRR | 1 / rank of the first relevant chunk (0 if none within max K), averaged over queries |
| NDCG@K | DCG@K / IDCG@K, with binary gains and a log2(rank + 1) discount |
| **Evidence Recall@K** | fraction of an item's evidence spans with at least one covering chunk in the top-K |
| Doc Recall@K | fraction of relevant documents among the first K distinct retrieved documents |

**Why Evidence Recall is the primary recall metric.** With overlapping chunks, one span
can be covered by two chunks. A system that retrieves just one of them has all the
evidence but only 50% chunk recall. Evidence Recall counts what matters: whether each
piece of required evidence was found. For multi-hop items it is exactly "did we find both
halves".

Unanswerable items have no evidence, so they are excluded from retrieval metrics and
reported as excluded. They are scored in generation evaluation, on whether the system
abstains.

All metric functions are unit-tested against hand-computed values
([tests/evaluation/test_retrieval_metrics.py](../tests/evaluation/test_retrieval_metrics.py)).

## Baseline results: dense retrieval (test split)

Run `20260928T080636-retrieval-dense-test` (report in
[data/experiments/](../data/experiments/)), dataset version `c6009ef85e53`, bge-small
embeddings with the `Title > Section` prefix, cosine HNSW. 198 test items with evidence
were evaluated (21 unanswerable excluded, 0 unmapped evidence spans).

| Metric | Value | 95% bootstrap CI |
|---|---:|---:|
| Evidence Recall@5 | **0.701** | 0.641 – 0.759 |
| Evidence Recall@10 | 0.761 | 0.710 – 0.815 |
| MRR | 0.624 | 0.573 – 0.682 |
| NDCG@10 | 0.595 | 0.548 – 0.646 |
| Chunk Recall@10 | 0.711 | |
| Doc Recall@5 | 0.838 | |
| Retrieval latency, p50 / p95 | 24 ms / 32 ms | |

By question type (Evidence Recall):

| Type | n | @1 | @5 | @10 | @20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| keyword_heavy | 31 | 0.565 | **0.871** | 0.871 | 0.903 | 0.707 |
| adversarial | 17 | 0.627 | 0.824 | 0.882 | 0.882 | 0.734 |
| simple_factual | 42 | 0.559 | 0.821 | 0.845 | 0.905 | 0.678 |
| semantic | 31 | 0.387 | 0.672 | 0.704 | 0.720 | 0.531 |
| multi_hop | 26 | 0.330 | 0.641 | 0.724 | 0.817 | 0.779 |
| numerical | 21 | 0.381 | 0.571 | 0.714 | 0.905 | 0.496 |
| ambiguous | 17 | 0.235 | **0.471** | 0.471 | 0.529 | 0.327 |
| comparison | 13 | 0.264 | 0.449 | 0.737 | 0.924 | 0.610 |

How to read these (small per-type n, so these are directions, not conclusions):

- **Ambiguous questions are the weakest (0.47 at 5, barely improving by 20).** Vague
  questions are not a ranking problem; the right evidence is simply not retrieved. This
  is what query rewriting targets.
- **Comparison and numerical gain a lot between K=5 and K=20** (0.45 → 0.92, 0.57 → 0.91).
  The evidence is retrieved but ranked low, which is what a cross-encoder reranker over a
  deeper candidate list targets.
- **Multi-hop has the second-highest MRR (0.78) but only 0.64 Evidence Recall@5.** The
  first half of the evidence is found early and the second often is not. This shows why
  Evidence Recall, not MRR, is the right lens for multi-hop questions.
- **Semantic (0.67) is below simple factual (0.82)**, as expected when the question avoids
  the passage's wording.
- Keyword-heavy questions are already strong for dense retrieval here (0.87). This is
  partly the lexical bias noted above; BM25 will show whether exact matching adds more.

## BM25 vs dense (test split)

Run `20260928T083304-retrieval-bm25-test`: words + compound analyzer, k1 = 1.2, b = 0.75.
All of these choices were made on dev (see [retrieval.md](retrieval.md#bm25)). Both
retrievers answer the same 198 questions, so differences are **paired**: bootstrapped per
query with `scripts/compare_retrieval_runs.py`.

| Metric | Dense | BM25 | Difference, BM25 − dense (95% CI) | Queries BM25 better / worse / tied |
|---|---:|---:|---:|---:|
| Evidence Recall@5 | 0.701 | **0.791** | **+0.090** [+0.034, +0.143] | 36 / 13 / 149 |
| Evidence Recall@10 | 0.761 | **0.844** | **+0.083** [+0.030, +0.134] | 31 / 14 / 153 |
| MRR | 0.624 | **0.725** | **+0.101** [+0.046, +0.155] | 66 / 34 / 98 |
| NDCG@10 | 0.595 | **0.715** | **+0.120** [+0.071, +0.164] | 87 / 37 / 74 |
| Latency p50 | 24 ms | 3.9 ms | | |

BM25's own 95% CI for Evidence Recall@5 is 0.740–0.842 and for MRR 0.671–0.776.

By question type (Evidence Recall@5):

| Type | n | Dense | BM25 | Difference |
|---|---:|---:|---:|---:|
| numerical | 21 | 0.571 | 0.952 | **+0.381** |
| adversarial | 17 | 0.824 | 1.000 | +0.176 |
| simple_factual | 42 | 0.821 | 0.905 | +0.083 |
| keyword_heavy | 31 | 0.871 | 0.935 | +0.065 |
| multi_hop | 26 | 0.641 | 0.689 | +0.048 |
| semantic | 31 | 0.672 | 0.710 | +0.038 |
| ambiguous | 17 | 0.471 | 0.471 | 0.000 |
| comparison | 13 | 0.449 | 0.368 | **−0.081** |

How to read this:

- **Part of BM25's lead is an artifact of the dataset.** Adversarial questions are
  generated from a fact statement written in the passage's own words, so they
  near-quote the evidence, and BM25 finds all of them (1.00). Numerical questions share
  exact numbers and units with their evidence. This is the lexical bias listed under
  [Known weaknesses](#known-weaknesses-of-the-dataset). The gap on semantic questions
  (+0.04), which were filtered for low word overlap, is much smaller and is a fairer
  indication of the difference on paraphrased queries.
- **Dense retrieval is better for comparison questions** (0.45 vs 0.37 at 5; 0.92 vs 0.68
  at 20). They are phrased conceptually ("how does X differ from Y"), which suits
  embeddings.
- **Neither helps ambiguous questions (0.47 each).** Vague questions lack the words that
  either method needs, and only query rewriting addresses that.
- **The two methods fail on different questions**: 36 queries where BM25 is better at 5
  and 13 where dense is better. That complementarity is the motivation for hybrid fusion
  (next phase), which will be measured the same way.

## Hybrid vs BM25 and dense (test split)

Run `20261003T110041-retrieval-hybrid-test`: linear fusion, weights 0.5 / 0.5, 50
candidates per retriever, chosen on dev. Paired differences:

| Metric | Dense | BM25 | Hybrid | Hybrid − dense (95% CI) | Hybrid − BM25 (95% CI) |
|---|---:|---:|---:|---:|---:|
| Evidence Recall@5 | 0.701 | 0.791 | **0.802** | **+0.101** [+0.056, +0.146] | +0.011 [−0.024, +0.048] |
| Evidence Recall@10 | 0.761 | 0.844 | **0.859** | **+0.098** [+0.056, +0.140] | +0.015 [−0.018, +0.048] |
| MRR | 0.624 | **0.725** | 0.723 | **+0.099** [+0.059, +0.139] | −0.002 [−0.034, +0.029] |
| NDCG@10 | 0.595 | **0.715** | 0.714 | **+0.119** [+0.085, +0.150] | −0.001 [−0.025, +0.024] |
| Latency p50 | 24 ms | 3.9 ms | 39 ms | | |

Hybrid's own 95% CI for Evidence Recall@5 is 0.750–0.852. Its Evidence Recall@20 is
0.909, the highest of the three.

By question type (Evidence Recall@5):

| Type | n | Dense | BM25 | Hybrid |
|---|---:|---:|---:|---:|
| semantic | 31 | 0.672 | 0.710 | **0.806** |
| multi_hop | 26 | 0.641 | 0.689 | **0.760** |
| comparison | 13 | 0.449 | 0.368 | **0.468** |
| keyword_heavy | 31 | 0.871 | **0.935** | **0.935** |
| adversarial | 17 | 0.824 | **1.000** | **1.000** |
| ambiguous | 17 | 0.471 | 0.471 | 0.471 |
| simple_factual | 42 | 0.821 | **0.905** | 0.881 |
| numerical | 21 | 0.571 | **0.952** | 0.809 |

How to read this:

- **Overall, hybrid is indistinguishable from BM25 on this dataset**, while costing about
  10× the latency. The tie is not a sign that hybrid adds nothing; it is shaped by the
  dataset's lexical bias. Hybrid **gains where questions are least lexical**: semantic
  +0.097, multi-hop +0.071, comparison +0.10 over BM25. It **loses where BM25's exact
  matching is decisive**: numerical −0.143, simple factual −0.024.
- It is best or tied-best on 6 of 8 types, so it is the most *robust* of the three. On
  real user queries, which are typically less lexically aligned with the docs than these
  passage-derived questions, the semantic-type result is the more relevant signal. That
  is an expectation, not a measurement.
- **Ambiguous questions are still at 0.47** for all three methods, as expected: fusion
  combines retrievers, but neither can find evidence for a question that lacks the key
  terms.
- **Hybrid has the best recall at depth (0.909 at 20).** That makes it the natural
  candidate generator for the reranker in the next phase, which re-orders a deeper list.

## Reranking (test split)

Run `20261006T183435-retrieval-hybrid_rerank-test`: hybrid top 20, reranked by
`BAAI/bge-reranker-base` on MPS. Depth was chosen on dev (see
[retrieval.md](retrieval.md#cross-encoder-reranking)). Paired differences:

| Metric | Hybrid | Hybrid + rerank | vs hybrid (95% CI) | vs BM25 (95% CI) |
|---|---:|---:|---:|---:|
| Evidence Recall@5 | 0.802 | **0.837** | **+0.034** [+0.003, +0.069] | **+0.045** [+0.010, +0.083] |
| Evidence Recall@10 | 0.859 | **0.893** | **+0.034** [+0.008, +0.064] | **+0.049** [+0.013, +0.087] |
| MRR | 0.723 | **0.760** | +0.037 [−0.004, +0.077] | +0.035 [−0.007, +0.079] |
| NDCG@10 | 0.714 | **0.758** | **+0.044** [+0.013, +0.074] | **+0.043** [+0.011, +0.077] |
| Latency p50 | 39 ms | 1.96 s | | |

The reranked configuration's own 95% CI is 0.789–0.882 for Evidence Recall@5 and
0.712–0.810 for MRR. Its Evidence Recall@20 equals hybrid's (0.909) by construction: the
reranker only re-orders hybrid's top 20.

By question type (Evidence Recall@5):

| Type | n | Dense | BM25 | Hybrid | Hybrid + rerank |
|---|---:|---:|---:|---:|---:|
| keyword_heavy | 31 | 0.871 | 0.935 | 0.935 | **1.000** |
| simple_factual | 42 | 0.821 | 0.905 | 0.881 | **0.929** |
| numerical | 21 | 0.571 | **0.952** | 0.809 | 0.905 |
| adversarial | 17 | 0.824 | **1.000** | **1.000** | 0.941 |
| semantic | 31 | 0.672 | 0.710 | 0.806 | **0.839** |
| multi_hop | 26 | 0.641 | 0.689 | **0.760** | 0.718 |
| comparison | 13 | 0.449 | 0.368 | 0.468 | **0.537** |
| ambiguous | 17 | 0.471 | 0.471 | 0.471 | **0.529** |

How to read this:

- **The first configuration to beat BM25 significantly** (Evidence Recall@5 +0.045, CI
  [+0.010, +0.083]; NDCG@10 +0.043). It is best on 6 of 8 question types.
- **It recovers most of hybrid's numerical loss** (0.81 → 0.91), and it moves ambiguous
  questions off 0.47 for the first time (0.53; MRR +0.15 over hybrid). A cross-encoder can
  sometimes match a vague question to the right passage when the question shares no
  distinctive words with it.
- **It slightly hurts multi-hop (−0.042) and adversarial (−0.059) questions.** A
  cross-encoder scores each passage independently against the whole question. For a
  two-part question, the passage covering one half can lose to a passage that looks
  relevant to all of it. False-premise questions contain a wrong number, and the reranker
  may prefer passages that echo it.
- **Dev and test disagree on which metric moves.** On dev, MRR improved significantly
  and Evidence Recall@5 did not. On test it is the reverse. Both splits move in the same
  direction, but which metric crosses significance depends on sampling noise at these
  sizes (88 dev, 198 test). This is a reminder not to over-read a single metric.
- **The cost is real:** about 2 s per query on MPS (50× hybrid), for +0.034 Evidence
  Recall@5. Whether that is worth it depends on the latency budget. Phase 14's experiment
  runner will put quality, latency and cost side by side for every configuration.

## Running

```bash
python scripts/create_eval_dataset.py                    # needs the LLM server
python scripts/run_retrieval_eval.py --retriever dense --split test
python scripts/run_retrieval_eval.py --retriever bm25 --split test
python scripts/run_retrieval_eval.py --retriever hybrid --split test
RERANKER_DEVICE=mps python scripts/run_retrieval_eval.py --retriever hybrid_rerank --split test
python scripts/compare_retrieval_runs.py data/experiments/<dense-run> data/experiments/<bm25-run>
python scripts/export_review_sheet.py --per-type 5       # CSV for human spot checks
```

Each run writes `data/experiments/<run-id>/report.json` (committed: config snapshot,
dataset version, overall and per-type metrics, latency) and `per_query.jsonl` (gitignored:
rankings and per-query metrics).
