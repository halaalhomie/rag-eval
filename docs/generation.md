# Generation and LLM providers

## Provider abstraction

Pipelines depend on one method ([app/generation/llm.py](../app/generation/llm.py)):

```python
LLMClient.complete(messages, *, model=None, max_tokens=None, temperature=None) -> LLMResponse
```

`LLMResponse` carries the text, the model, input and output token counts, latency and
attempts. Providers map transport failures onto one error hierarchy, so retry policy and
pipeline error handling never contain provider-specific code.

| Provider | Covers |
|---|---|
| `openai_compatible` | any server speaking OpenAI chat-completions: local **MLX** (default), Ollama, vLLM, llama.cpp; hosted OpenAI, Groq, Gemini, Hugging Face Inference Providers |
| `fake` | deterministic canned or scripted replies, for tests and offline development |

Switching models or providers is configuration only: `OPENAI_BASE_URL`, `LLM_MODEL` and
optionally `OPENAI_API_KEY`.

### Retries and failures

[app/generation/openai_compatible.py](../app/generation/openai_compatible.py) talks HTTP
directly with httpx. The wire format is small, and owning it keeps the failure behaviour
explicit:

| Failure | Behaviour |
|---|---|
| timeout, connection error, HTTP 5xx, malformed body | retried with exponential backoff and full jitter (capped at 30 s) |
| HTTP 429 | retried, honoring `Retry-After` |
| other HTTP 4xx | **not retried**: the request is wrong, and retrying cannot help |
| all attempts used (`LLM_MAX_RETRIES`, default 3) | error raised; `/query` returns 502 |

### Structured output

Graders and analyzers need JSON. Many OpenAI-compatible servers, including
`mlx_lm.server`, do not enforce JSON schemas, so
[app/generation/structured.py](../app/generation/structured.py) handles it identically
for every provider:

1. The Pydantic schema is described in the system prompt.
2. The reply is parsed tolerantly: code fences are stripped, and the first balanced JSON
   object is taken from any surrounding prose, with braces inside strings handled.
3. The validation error is fed back to the model for a bounded number of repair attempts
   (default 1).
4. After that, `StructuredOutputError` is raised. Nothing silently substitutes a default
   score; callers must choose an explicit fallback.

### Cost accounting

Each call is recorded with a purpose (`generate`, and later `grade`, `rewrite`, `judge`).
Cost uses `LLM_PRICING` (USD per 1M input and output tokens). Models in `LLM_LOCAL_MODELS`
cost $0 per token. Any other model is reported under **`unpriced_models`** rather than
silently counted as free, so a paid model with a missing price can't make a strategy
look cheaper than it is.

## Default local model

| | |
|---|---|
| Model | `mlx-community/Qwen2.5-3B-Instruct-4bit` (Hugging Face), about 1.7 GB |
| Runtime | [`mlx-lm`](https://github.com/ml-explore/mlx-lm) on the Apple GPU (Metal), via `mlx_lm.server` |
| Why | free, offline, fits in 8 GB next to Postgres and the embedder; the OpenAI-compatible server needs no custom client code |

```bash
pip install -e ".[local-llm]"
mlx_lm.server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8080
```

MLX runs natively on macOS and cannot run inside the Linux API container, which has no
access to the Apple GPU. `docker compose` points the API at the host
(`http://host.docker.internal:8080/v1`).

### Measured behaviour (dev machine, not a benchmark)

- Structured JSON probe: **6/6** grading replies were valid JSON matching the requested
  schema, at about 1.1 s each once warm.
- Baseline answers, top-5 contexts (about 1.6–1.8k prompt tokens): generation takes about
  **15–19 s**, dominated by prompt processing. A 300-question baseline run is on the order
  of 1.5 hours; multi-call strategies scale with their number of LLM calls.

### Known limitation: a 3B model is a weak grader

In the JSON probe, one document stated the answer outright ("all deletes are graceful
within 30 seconds"). The model still returned `contains_answer: false` for it on all 3
repeats. Formatting is reliable; judgment is not. So:

- document-relevance grading for Corrective RAG will use a **cross-encoder** as the
  primary grader, with the LLM grader kept as a switchable comparison
- claim verification will use **MiniCheck-Flan-T5-Large**, a small model trained for
  checking claims against grounding documents
- all generation-quality numbers will name the generator and judge models. They compare
  strategies against each other *under that model*, and the same experiments can be
  re-run with a stronger model by changing configuration only

## Answer prompt and citations

One prompt is shared by every strategy
([app/rag/prompts/generation.py](../app/rag/prompts/generation.py)), so strategies differ
only in what they retrieve. Contexts are numbered `[1]..[n]`, and the model must cite
inline and use an exact abstention phrase ("I don't know based on the provided sources.")
when the sources lack the answer. That makes abstention detectable deterministically,
without a judge.

[app/generation/citations.py](../app/generation/citations.py) maps each `[n]` back to its
chunk. The API returns structured citations with the chunk ID, document, section, page,
URL and character span. Numbers that match no provided source are returned as
`invalid_citations`, a deterministic signal of citation hallucination.

### Prompt iteration (8-question probe)

The first prompt put all rules in the system message. Qwen 3B frequently ignored the
citation rule. The probe measured answers with at least one valid citation (top-5
contexts):

| Prompt | Answers with valid citations | Answers that talk about "the sources" |
|---|---:|---:|
| Rules in the system prompt | 1/8 | 2/8 |
| + format example | 2/8 | 0/8 |
| + citation rule restated after the sources | **8/8** | 0/8 |

Small models lose system-prompt instructions behind about 1.5k tokens of context, so the
rule is repeated next to the question. This is an 8-question sanity probe used to pick a
reasonable baseline prompt, not a benchmark result. Citation correctness is measured
properly by the evaluation framework.

## Baseline strategy

[app/rag/strategies/baseline.py](../app/rag/strategies/baseline.py):
embed query → dense top-k → one LLM call → parse citations.

- No rewriting, reranking or grading, and exactly one LLM call. It is the reference point
  every other strategy is measured against.
- Empty retrieval makes it **abstain without calling the LLM**: there is nothing to
  ground an answer in, and the call would cost tokens for a guaranteed hallucination risk.
- Timings for retrieval, generation and total, plus token usage, are returned with every
  answer.

Strategies are registered in [app/rag/strategies/\_\_init\_\_.py](../app/rag/strategies/__init__.py)
only once they are implemented. Requesting an unimplemented strategy returns HTTP 501
rather than silently falling back to another pipeline, which would corrupt experiment
comparisons.
