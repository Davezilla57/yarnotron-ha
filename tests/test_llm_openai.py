"""The OpenAI-compatible model client (LiteLLM, vLLM, llama.cpp, ...) and the settings around it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from yarnotron.core.config import Settings
from yarnotron.services.llm import build_llm_client
from yarnotron.services.llm_openai import OpenAICompatClient
from yarnotron.services.memory import (
    OllamaEmbedder,
    deterministic_embedding,
    embed_document,
    embed_query,
)
from yarnotron.services.ollama import (
    OllamaClient,
    OllamaMalformedJsonError,
    OllamaUnavailableError,
    get_ollama_client,
)

BASE = "http://llm.test:4000/v1"
FENCED = (
    '```json\n{"response_text": "Oh, a seal pup! What does Wren do?", '
    '"story_patches": [{"operation": "add_scene"}], "tutor_notes": []}\n```'
)
PLAIN = '{"response_text": "Hello there.", "story_patches": []}'


def settings(**overrides) -> Settings:
    values = {"llm_backend": "openai", "llm_base_url": BASE}
    values.update(overrides)
    return Settings(**values)


def chat_reply(content: str | None) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class Recorder:
    """A MockTransport that remembers requests and answers with `responder(request)`."""

    def __init__(self, responder):
        self.requests: list[httpx.Request] = []
        self._responder = responder
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)

    @property
    def last_json(self) -> dict:
        return json.loads(self.requests[-1].content)


def client_with(responder, **setting_overrides) -> tuple[OpenAICompatClient, Recorder]:
    recorder = Recorder(responder)
    client = OpenAICompatClient(
        settings(**setting_overrides), client=httpx.AsyncClient(transport=recorder.transport)
    )
    return client, recorder


def sse_body(content: str, chunk_size: int = 7, usage: dict | None = None) -> str:
    pieces = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    lines = [": keep-alive", ""]
    for piece in pieces:
        event = {"choices": [{"delta": {"content": piece}, "finish_reason": None}]}
        lines += [f"data: {json.dumps(event)}", ""]
    lines += [f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}", ""]
    if usage is not None:
        lines += [f"data: {json.dumps({'choices': [], 'usage': usage})}", ""]
    lines += ["data: [DONE]", ""]
    return "\n".join(lines)


def streaming_client(content: str, chunk_size: int = 7, usage: dict | None = None, **overrides):
    return client_with(
        lambda request: httpx.Response(200, text=sse_body(content, chunk_size, usage)), **overrides
    )


async def collect(client: OpenAICompatClient):
    texts, payload, stats = [], None, None
    async for event in client.stream_json("prompt"):
        if event.text is not None:
            texts.append(event.text)
        if event.payload is not None:
            payload, stats = event.payload, event.stats
    return "".join(texts), payload, stats


# ---------------------------------------------------------------------- settings + factory


def test_openai_backend_requires_a_base_url():
    with pytest.raises(ValidationError):
        Settings(llm_backend="openai")
    with pytest.raises(ValidationError):
        Settings(llm_backend="openai", llm_base_url="   ")


def test_default_backend_is_ollama_and_blank_key_means_none():
    default = Settings()
    assert default.llm_backend == "ollama"
    assert Settings(llm_api_key="  ").llm_api_key is None


def test_factory_picks_the_client_from_settings():
    assert type(build_llm_client(Settings())) is OllamaClient
    assert type(build_llm_client(settings())) is OpenAICompatClient


async def test_dependency_fallback_uses_the_configured_backend():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert isinstance(get_ollama_client(request, settings()), OpenAICompatClient)
    request.app.state.ollama_client = "already-built"
    assert get_ollama_client(request, settings()) == "already-built"


# ---------------------------------------------------------------------- request shape


async def test_generate_json_request_matches_the_ollama_contract():
    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply('{"ok": true}')))
    result = await client.generate_json("Return JSON.", model="yarn-story")

    assert result == {"ok": True}
    assert str(rec.requests[0].url) == f"{BASE}/chat/completions"
    assert rec.last_json == {
        "model": "yarn-story",
        "messages": [{"role": "user", "content": "Return JSON."}],
        "stream": False,
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }


async def test_options_are_mapped_and_unknown_ones_are_not_forwarded():
    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")))
    await client.generate_json("p", options={"num_predict": 160, "mirostat": 2})
    body = rec.last_json
    assert body["max_tokens"] == 160
    assert "mirostat" not in body and "num_predict" not in body


async def test_seed_is_forwarded_when_set():
    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")), model_seed=42)
    await client.generate_json("p")
    assert rec.last_json["seed"] == 42


async def test_json_mode_and_thinking_hint_are_opt_in_and_opt_out():
    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")))
    await client.generate_json("p")
    assert "response_format" not in rec.last_json

    client, rec = client_with(
        lambda r: httpx.Response(200, json=chat_reply("{}")),
        llm_json_mode=True,
        llm_thinking_hint=False,
    )
    await client.generate_json("p")
    assert rec.last_json["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in rec.last_json and "reasoning_effort" not in rec.last_json

    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")))
    await client.generate_json("p", think=None)
    assert "chat_template_kwargs" not in rec.last_json and "reasoning_effort" not in rec.last_json

    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")))
    await client.generate_json("p", think=True)
    assert rec.last_json["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in rec.last_json


async def test_bearer_token_is_sent_only_when_configured():
    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")), llm_api_key="sk-x")
    await client.generate_json("p")
    assert rec.requests[0].headers["authorization"] == "Bearer sk-x"

    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("{}")))
    await client.generate_json("p")
    assert "authorization" not in rec.requests[0].headers


async def test_generate_uses_no_default_options():
    client, rec = client_with(lambda r: httpx.Response(200, json=chat_reply("  A sentence. ")))
    assert await client.generate("Describe a gull.") == "A sentence."
    assert "max_tokens" not in rec.last_json and "temperature" not in rec.last_json


# ---------------------------------------------------------------------- reply parsing


async def test_fenced_and_prose_wrapped_json_is_recovered():
    client, _ = client_with(lambda r: httpx.Response(200, json=chat_reply(FENCED)))
    assert (await client.generate_json("p"))["story_patches"] == [{"operation": "add_scene"}]
    client, _ = client_with(
        lambda r: httpx.Response(200, json=chat_reply('Sure! {"action": "allow"} Hope that helps.'))
    )
    assert await client.generate_json("p") == {"action": "allow"}


@pytest.mark.parametrize(
    "content", ["no json at all", "[1, 2, 3]", "", None, '{"unterminated": ']
)
async def test_unreadable_replies_are_malformed_not_unavailable(content):
    client, _ = client_with(lambda r: httpx.Response(200, json=chat_reply(content)))
    with pytest.raises(OllamaMalformedJsonError):
        await client.generate_json("p")


async def test_no_choices_is_malformed():
    client, _ = client_with(lambda r: httpx.Response(200, json={"choices": []}))
    with pytest.raises(OllamaMalformedJsonError):
        await client.generate_json("p")


# ---------------------------------------------------------------------- failures keep the Ollama contract


@pytest.mark.parametrize("status", [401, 404, 429, 500, 503])
async def test_http_errors_are_unavailable(status):
    client, _ = client_with(lambda r: httpx.Response(status, json={"error": "nope"}))
    with pytest.raises(OllamaUnavailableError) as info:
        await client.generate_json("p")
    assert BASE in str(info.value)
    with pytest.raises(OllamaUnavailableError):
        await client.list_models()
    with pytest.raises(OllamaUnavailableError):
        await client.embed("x", "yarn-embed")
    with pytest.raises(OllamaUnavailableError):
        await collect(client)


async def test_connection_failure_is_unavailable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, _ = client_with(refuse)
    with pytest.raises(OllamaUnavailableError):
        await client.generate_json("p")
    with pytest.raises(OllamaUnavailableError):
        await collect(client)


# ---------------------------------------------------------------------- models + embeddings


async def test_list_models_reads_openai_model_ids():
    client, rec = client_with(
        lambda r: httpx.Response(
            200, json={"data": [{"id": "yarn-story"}, {"id": "yarn-safety"}, {"object": "x"}]}
        )
    )
    assert [m.name for m in await client.list_models()] == ["yarn-story", "yarn-safety"]
    assert str(rec.requests[0].url) == f"{BASE}/models"


async def test_embed_returns_the_vector():
    client, rec = client_with(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, -0.25, 1]}]})
    )
    assert await client.embed("hello", "yarn-embed") == [0.5, -0.25, 1.0]
    assert str(rec.requests[0].url) == f"{BASE}/embeddings"
    assert rec.last_json == {"model": "yarn-embed", "input": "hello"}


@pytest.mark.parametrize("payload", [{"data": []}, {"data": [{"embedding": []}]}, {}])
async def test_embed_without_a_vector_is_unavailable(payload):
    client, _ = client_with(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(OllamaUnavailableError):
        await client.embed("hello", "yarn-embed")


# ---------------------------------------------------------------------- streaming


async def test_streams_text_then_final_payload():
    client, rec = streaming_client(PLAIN)
    text, payload, _ = await collect(client)
    assert text == "Hello there."
    assert payload == {"response_text": "Hello there.", "story_patches": []}
    body = rec.last_json
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


async def test_streams_through_a_json_fence():
    client, _ = streaming_client(FENCED)
    text, payload, _ = await collect(client)
    assert text == "Oh, a seal pup! What does Wren do?"
    assert payload["story_patches"] == [{"operation": "add_scene"}]


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50, 500])
async def test_text_is_identical_at_any_chunk_size(chunk_size):
    client, _ = streaming_client(FENCED, chunk_size)
    text, payload, _ = await collect(client)
    assert text == "Oh, a seal pup! What does Wren do?"
    assert payload is not None


async def test_field_done_fires_before_the_payload():
    client, _ = streaming_client(FENCED, chunk_size=3)
    events = [e async for e in client.stream_json("prompt")]
    done_at = next(i for i, e in enumerate(events) if e.field_done)
    payload_at = next(i for i, e in enumerate(events) if e.payload is not None)
    assert done_at < payload_at
    assert sum(1 for e in events if e.field_done) == 1
    spoken = "".join(e.text for e in events[: done_at + 1] if e.text)
    assert spoken == "Oh, a seal pup! What does Wren do?"
    assert payload_at == len(events) - 1


async def test_stats_come_from_usage_and_local_clocks():
    client, _ = streaming_client(PLAIN, usage={"prompt_tokens": 2100, "completion_tokens": 76})
    _, _, stats = await collect(client)
    assert stats["prompt_eval_count"] == 2100 and stats["eval_count"] == 76
    assert stats["load_duration"] == 0
    assert stats["prompt_eval_duration"] >= 0 and stats["eval_duration"] >= 0


async def test_stats_still_present_when_the_server_sends_no_usage():
    client, _ = streaming_client(PLAIN)
    _, _, stats = await collect(client)
    assert "prompt_eval_count" not in stats and "prompt_eval_duration" in stats


async def test_stream_that_never_forms_json_is_malformed():
    client, _ = streaming_client("I would rather not say.")
    with pytest.raises(OllamaMalformedJsonError):
        await collect(client)


# ---------------------------------------------------------------------- embedding prefixes


class FakeEmbedClient:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def embed(self, text: str, model: str) -> list[float]:
        self.calls.append((text, model))
        return [1.0, 0.0, 0.0, 0.0]


async def test_query_and_document_prefixes_are_applied_separately():
    fake = FakeEmbedClient()
    embedder = OllamaEmbedder(
        Settings(
            embedding_model="yarn-embed",
            embedding_dimensions=4,
            embedding_query_prefix="task: search result | query:",
            embedding_document_prefix="title: none | text:",
        ),
        fake,
    )
    await embed_query(embedder, "the door")
    await embed_document(embedder, "Wren found a door")
    await embedder.embed("plain embed is the document form")
    assert fake.calls == [
        ("task: search result | query: the door", "yarn-embed"),
        ("title: none | text: Wren found a door", "yarn-embed"),
        ("title: none | text: plain embed is the document form", "yarn-embed"),
    ]


async def test_no_prefix_configured_changes_nothing():
    fake = FakeEmbedClient()
    embedder = OllamaEmbedder(Settings(embedding_model="nomic-embed-text:v1.5", embedding_dimensions=4), fake)
    await embed_query(embedder, "the door")
    await embed_document(embedder, "a door")
    assert [c[0] for c in fake.calls] == ["the door", "a door"]


async def test_hash_fallback_ignores_prefixes():
    embedder = OllamaEmbedder(
        Settings(embedding_model=None, embedding_dimensions=16, embedding_query_prefix="q:"),
        FakeEmbedClient(),
    )
    assert await embed_query(embedder, "door") == deterministic_embedding("door", 16)


async def test_helpers_fall_back_to_plain_embed_for_simple_embedders():
    class Simple:
        async def embed(self, text: str) -> list[float]:
            return [float(len(text))]

    assert await embed_query(Simple(), "abc") == [3.0]
    assert await embed_document(Simple(), "abcd") == [4.0]
