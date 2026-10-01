import json

import httpx
import pytest

from yarnotron.core.config import Settings
from yarnotron.services.ollama import OllamaClient, OllamaUnavailableError

FENCED = (
    '```json\n{"response_text": "Oh, a seal pup! What does Wren do?", '
    '"story_patches": [{"operation": "add_scene"}], "tutor_notes": []}\n```'
)
PLAIN = '{"response_text": "Hello there.", "story_patches": []}'


def ndjson_transport(body: str, chunk_size: int = 7) -> httpx.MockTransport:
    """Serve `body` the way Ollama does: one JSON line per token-ish chunk."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        pieces = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
        lines = [json.dumps({"response": p, "done": False}) for p in pieces]
        lines.append(json.dumps({"response": "", "done": True}))
        return httpx.Response(200, text="\n".join(lines))

    return httpx.MockTransport(handler)


def client_for(body: str, chunk_size: int = 7) -> OllamaClient:
    settings = Settings(ollama_base_url="http://test")
    return OllamaClient(settings, client=httpx.AsyncClient(transport=ndjson_transport(body, chunk_size)))


async def collect(client: OllamaClient):
    texts, payload = [], None
    async for event in client.stream_json("prompt"):
        if event.text is not None:
            texts.append(event.text)
        if event.payload is not None:
            payload = event.payload
    return "".join(texts), payload


async def test_streams_text_then_final_payload():
    text, payload = await collect(client_for(PLAIN))
    assert text == "Hello there."
    assert payload == {"response_text": "Hello there.", "story_patches": []}


async def test_streams_through_a_json_fence():
    """gemma4 fences every draft, so this is the normal path, not an edge case."""
    text, payload = await collect(client_for(FENCED))
    assert text == "Oh, a seal pup! What does Wren do?"
    assert payload["story_patches"] == [{"operation": "add_scene"}]


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50, 500])
async def test_text_is_identical_at_any_chunk_size(chunk_size):
    text, payload = await collect(client_for(FENCED, chunk_size))
    assert text == "Oh, a seal pup! What does Wren do?"
    assert payload is not None


async def test_text_arrives_before_the_payload():
    """The whole point: speak the reply before the patches finish generating."""
    client = client_for(FENCED, chunk_size=3)
    order = []
    async for event in client.stream_json("prompt"):
        order.append("text" if event.text is not None else "payload")
    assert order[-1] == "payload"
    assert "text" in order[:-1]
    assert order.count("payload") == 1


async def test_field_done_fires_before_the_payload():
    """The closing quote of response_text is the cue to speak the last sentence.

    Without it the tail waits for the story patches to finish generating, which
    measured as a ~3 s silence gap mid-reply against real Ollama.
    """
    client = client_for(FENCED, chunk_size=3)
    events = [e async for e in client.stream_json("prompt")]

    done_at = next(i for i, e in enumerate(events) if e.field_done)
    payload_at = next(i for i, e in enumerate(events) if e.payload is not None)
    assert done_at < payload_at
    assert sum(1 for e in events if e.field_done) == 1
    # Text emitted up to and including field_done is the whole spoken reply.
    spoken = "".join(e.text for e in events[: done_at + 1] if e.text)
    assert spoken == "Oh, a seal pup! What does Wren do?"


async def test_unparseable_output_raises():
    client = client_for("not json at all")
    with pytest.raises(OllamaUnavailableError):
        await collect(client)


async def test_transport_failure_raises_unavailable():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    settings = Settings(ollama_base_url="http://test")
    client = OllamaClient(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    with pytest.raises(OllamaUnavailableError):
        await collect(client)


async def test_think_is_disabled_by_default():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=json.dumps({"response": PLAIN, "done": True}))

    settings = Settings(ollama_base_url="http://test")
    client = OllamaClient(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await collect(client)
    assert seen["think"] is False
    assert seen["stream"] is True
