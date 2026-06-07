# LightRAG — Added Features & Changelog

This document tracks features added on top of upstream LightRAG that are not part of the original project.

---

## 1. Langfuse Observability

**Files:** `lightrag/utils_langfuse.py`, `lightrag/llm/openai.py`, `lightrag/llm/ollama.py`

LightRAG can emit traces to [Langfuse](https://langfuse.com) for end-to-end observability of every LLM call and embedding call.

### How it works

- **OpenAI/compatible**: uses `langfuse.openai.AsyncOpenAI` — a drop-in replacement that auto-instruments all calls via the OpenAI SDK.
- **Ollama**: uses the `langfuse_observation` context manager from `utils_langfuse.py`, wrapping each chat and embed call manually. Captures prompt/response, token usage (`prompt_eval_count`, `eval_count`), and streaming output.
- **Privacy controls**: content capture is opt-in. When `LANGFUSE_CAPTURE_CONTENT=false` (the default), only character counts and token counts are sent — no prompt or response text leaves the system.

### Configuration

Add these to your `.env`:

```env
# Required
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted instance

# Enable tracing (default: disabled)
LANGFUSE_ENABLE_TRACE=true

# Optional: capture full prompt/response text in traces (default: false)
LANGFUSE_CAPTURE_CONTENT=false
LANGFUSE_MAX_CONTENT_CHARS=4000            # truncation limit when capture is on

# Alias: if you set LANGFUSE_BASE_URL instead of LANGFUSE_HOST, it is mapped automatically
```

### What appears in Langfuse

| Observation name | Type | Captured |
|---|---|---|
| `lightrag.ollama.chat` | generation | model, prompt chars, usage tokens, output |
| `lightrag.ollama.embed` | embedding | model, text count, embedding count |
| OpenAI calls | generation (auto) | all fields via SDK instrumentation |

### Notes

- Tracing is entirely optional and zero-cost when `LANGFUSE_ENABLE_TRACE` is not set.
- The Langfuse client is initialized lazily (once, cached) — no startup penalty.
- `utils_langfuse.py` is the single source of truth for the `LANGFUSE_HOST`/`LANGFUSE_BASE_URL` alias logic.

---

## 2. Retrieval-Based Conversation Memory

**Files:** `lightrag/lightrag.py`, `lightrag/prompt.py`, `lightrag/operate.py`

**Test:** `tests/test_memory_persistence.py`

Instead of sending the raw full conversation history to the LLM on every call, LightRAG persists conversation turns to disk and retrieves only the *semantically relevant* past messages for each new query. This enables recall of facts stated in previous browser sessions.

### How it works

```text
Session A                          Session B (new browser session)
─────────────────────────────      ─────────────────────────────────────────
User: "AquaVec produces 384-dim"   User: "What dims does our model produce?"
  │                                  │
  ├─ embed & save to memory.json     ├─ embed current query
  │                                  ├─ cosine-search memory.json
  └─ LLM responds, save response     ├─ retrieve: "AquaVec produces 384-dim"
                                     ├─ prepend facts to query text  ←── key step
                                     └─ LLM sees the fact → answers "384"
```

**Step-by-step per query:**

1. **Persist prior history** — any `conversation_history` passed by the caller (both user and assistant turns) is embedded and saved to `conversation_memory.json`. Deduplication is by content hash so repeated calls don't inflate the store.

2. **Retrieve** — the current query is embedded and compared (cosine similarity) against all stored messages. Messages above `memory_similarity_threshold` (default 0.4) are kept, top-k returned.

3. **Inject** — retrieved facts are injected in two places:
   - Prepended to the **query text itself** as a `[Prior conversation context]` block. This is the primary injection path and works reliably on small models (7B) that ignore `history_messages` and `user_prompt` additions.
   - Also prepended to **`conversation_history`** so the LLM sees them as chat turns for models that do handle history.

4. **Save current query** — after retrieval (so no self-retrieval in this turn), the current user query is saved to memory. This is the only path that captures single-turn sessions where the user states a fact and then clears the chat.

5. **Save assistant response (streaming and non-streaming)** — the assistant response is saved to memory after the LLM responds. For non-streaming queries the content is available immediately. For streaming queries, the response_iterator is wrapped in a collecting generator (`_collecting_stream`) that yields each chunk to the browser unchanged while simultaneously appending it to a buffer; the full assembled response is saved to memory after the last chunk is sent. No added latency to the stream.

### Why query-text injection (not just history_messages)

Qwen 2.5:7b and other small models reliably ignore chat-turn history when the system prompt contains a hard grounding constraint (`ONLY use the Context`). Three injection strategies were tested:

| Strategy | Outcome |
|---|---|
| `history_messages` only | Ignored by 7B models |
| `user_prompt` addition (appended to system prompt) | Ignored — `ONLY Context` constraint wins |
| Prepend to query text | **Works** — models always read the user message |

### Prompt changes (`lightrag/prompt.py`)

`rag_response` and `naive_rag_response` were updated to:

- Remove the `ONLY` constraint from the role description (was: "answer by ONLY using the provided Context")
- Declare conversation history as a **first-class source of truth** alongside the Context
- Add instruction to check conversation history **first** before consulting the Knowledge Graph
- Update the cannot-answer condition: "cannot be found in either the Context **or** the conversation history"

### Cache fix (`lightrag/operate.py`)

The LLM response cache hash now includes `conversation_history` content. Without this, the same query returned a stale cached response regardless of what memory context was injected.

### Configuration

```python
rag = LightRAG(
    ...
    memory_retrieval_enabled=True,      # default: True
    memory_top_k=3,                     # past messages to retrieve per query
    memory_similarity_threshold=0.4,    # cosine threshold (0–1); lower = broader recall
)
```

To disable entirely:

```python
rag = LightRAG(..., memory_retrieval_enabled=False)
```

### Storage

Memory is stored in `<working_dir>[/<workspace>]/conversation_memory.json`. Format:

```json
{
  "messages": [
    {
      "id": "memory-<md5hash>",
      "role": "user",
      "content": "...",
      "created_at": 1748649600,
      "embedding": [0.12, -0.34, ...]
    }
  ]
}
```

- Maximum 5000 messages retained (oldest are dropped when limit is exceeded).
- Concurrent access is guarded by an `asyncio.Lock` — safe under multi-request API server use.
- Messages without embeddings (e.g., from older versions) are skipped during retrieval.

### Does it work for multi-turn within the same session?

**Yes — via two separate mechanisms:**

| Scenario | Mechanism |
|---|---|
| Same session, UI sends history | Normal `conversation_history` flow — the UI passes prior turns directly in `QueryParam.conversation_history`; the LLM sees the full context natively. Memory retrieval also fires but is mostly redundant here. |
| Same session, programmatic use without explicit history | Memory retrieval fills the gap — Turn 1 facts are in `conversation_memory.json` by Turn 2. |
| Cross-session (new browser tab / cleared chat) | Memory retrieval is the only mechanism. The UI sends empty `conversation_history`; retrieved facts are the sole context source. |

The native `conversation_history` path is more direct for same-session use. Memory retrieval is most critical for the **cross-session** case where no history is sent.

### Known limitations

| Limitation | Notes |
|---|---|
| No per-user / per-session isolation | All conversations share one `conversation_memory.json`. A single-user deployment is fine; multi-user requires namespacing. |
| O(n) retrieval scan | Memory retrieval does a linear scan over all stored embeddings loaded from JSON. Adequate up to several thousand messages; a vector DB backend would be needed at scale. |
| Wrong answers from cold-start sessions | If memory is empty and the LLM answers incorrectly, that wrong answer is saved. Future retrievals may surface it alongside correct facts. Mitigated by the prompt change (LLM now uses history facts directly) but not eliminated. |

### Future improvements

- Move memory into LightRAG's existing vector store backend (removes O(n) scan)
- Per-session or per-user memory namespacing for multi-user deployments
- Configurable `memory_similarity_threshold` via `.env`

---

## 3. `--reload` flag for `lightrag-server`

**File:** `lightrag/api/lightrag_server.py`

The development server now supports `--reload` for automatic code reloading on file changes:

```bash
lightrag-server --reload
```

When `--reload` is set, the app is passed to uvicorn as the `get_application` factory string (required by uvicorn's reloader). In non-reload mode the app object is passed directly (original behaviour, no change).

**Important:** Without `--reload`, restarting the server manually is required after any code change. Code changes do not take effect in a running process.

---

## 4. RAGAS Evaluation (planned)

**Status:** Infrastructure in place; evaluation harness not yet implemented.

[RAGAS](https://docs.ragas.io) measures RAG quality via metrics like **faithfulness**, **answer relevancy**, **context precision**, and **context recall** — all without human-labelled ground truth.

### What's already available

- `aquery_llm` returns a structured dict containing both the LLM answer and the retrieved context, which is the exact input RAGAS expects.
- Langfuse traces capture prompt + retrieved context per query, making it possible to export a trace dataset directly into RAGAS.

### What needs to be built

1. **Context extractor** — a utility that runs `aquery_data` to get the raw retrieved chunks (entities, relations, text snippets) as a plain list of strings.

2. **Evaluation dataset builder** — a script that, given a list of test questions (+ optional ground-truth answers), runs LightRAG and collects `(question, answer, contexts)` tuples.

3. **RAGAS runner** — wires the dataset into `ragas.evaluate`:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

data = {
    "question": [...],
    "answer":   [...],
    "contexts": [[chunk1, chunk2], ...],   # list-of-lists
    "ground_truth": [...],                 # optional
}
result = evaluate(Dataset.from_dict(data), metrics=[faithfulness, answer_relevancy])
print(result)
```

1. **Langfuse → RAGAS bridge** (optional) — pull traces from Langfuse via `get_traces()` and feed them into RAGAS, avoiding a separate evaluation run.

### Installation

```bash
uv add ragas datasets
# RAGAS also requires an OpenAI key (used internally by its LLM-based metrics)
```
