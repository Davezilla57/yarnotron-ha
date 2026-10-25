# Model servers: Ollama or an OpenAI-compatible endpoint

Yarnotron's story model, safety reviewer, interest summaries and embeddings all go through one
client with five methods (`list_models`, `generate`, `generate_json`, `stream_json`, `embed`).
Two implementations exist; `YARNOTRON_LLM_BACKEND` picks one.

| Setting | Default | Meaning |
|---|---|---|
| `YARNOTRON_LLM_BACKEND` | `ollama` | `ollama` (native API) or `openai` (OpenAI-compatible) |
| `YARNOTRON_LLM_BASE_URL` | none | Required for `openai`. Include `/v1`, e.g. `http://127.0.0.1:4000/v1` |
| `YARNOTRON_LLM_API_KEY` | none | Sent as `Authorization: Bearer ...` when set |
| `YARNOTRON_LLM_JSON_MODE` | false | Also send `response_format: json_object`. Replies are parsed tolerantly either way (fences and prose around the object are stripped) |
| `YARNOTRON_LLM_THINKING_HINT` | true | Send `chat_template_kwargs.enable_thinking` (false for story and safety calls). Turn off for a strict endpoint that rejects unknown fields |
| `YARNOTRON_EMBEDDING_QUERY_PREFIX` / `..._DOCUMENT_PREFIX` | empty | Text put in front of a search query / a stored passage before embedding |

With `openai`, `YARNOTRON_DEFAULT_MODEL`, `YARNOTRON_SAFETY_MODEL` and `YARNOTRON_EMBEDDING_MODEL` are model or
alias names on that server. `YARNOTRON_OLLAMA_BASE_URL` is not used.

## What maps to what

| Ollama native | OpenAI-compatible |
|---|---|
| `POST /api/generate` (prompt) | `POST /chat/completions` (the prompt as one user message) |
| `options.num_predict` | `max_tokens` |
| `options.temperature / seed / top_p / stop` | the same fields |
| `think: false` | `chat_template_kwargs: {"enable_thinking": false}` (see `YARNOTRON_LLM_THINKING_HINT`) |
| NDJSON stream | server-sent events (`data: {...}` ... `data: [DONE]`) with `stream_options.include_usage` |
| `GET /api/tags` | `GET /models` |
| `POST /api/embed` | `POST /embeddings` |
| per-call counters (`prompt_eval_count` ...) | `usage` block plus local clocks: `prompt_eval_duration` = time to first token, `eval_duration` = time after it, `load_duration` = 0 |

Failures raise the same exceptions as the Ollama client (`OllamaUnavailableError`, `OllamaMalformedJsonError`), so the
routes still return 503 when the model server is down and the safety reviewer still retries unreadable output.
The `models.ollama_reachable` field in `/v1/status` means "the configured model server answered".

## Embedding prefixes

`nomic-embed-text` (the default) needs none. EmbeddingGemma expects `task: search result | query:` before a search
query and `title: none | text:` before a stored passage. Searches (`/v1/memory/search`, the per-turn memory lookup)
use the query prefix; stored memory chunks use the document prefix. Changing the embedding model or prefixes
requires re-embedding existing memories (a new database needs nothing). The hash fallback used when no embedding
model is set ignores the prefixes.

## Example: a LiteLLM proxy

```
YARNOTRON_LLM_BACKEND=openai
YARNOTRON_LLM_BASE_URL=http://127.0.0.1:4000/v1
YARNOTRON_LLM_API_KEY=<a virtual key limited to the three aliases below>
YARNOTRON_DEFAULT_MODEL=yarn-story
YARNOTRON_SAFETY_MODEL=yarn-safety
YARNOTRON_EMBEDDING_MODEL=yarn-embed
```

Keep the proxy's message logging off and every alias on local engines: the child's words pass through it.
