from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from yarnotron.agent.turn_processor import get_turn_processor
from yarnotron.api.routes import agent
from yarnotron.models.agent import AgentTurnResponse, TurnIntent
from yarnotron.models.safety import SafetyDecision
from yarnotron.services.ollama import OllamaUnavailableError


class ScriptedProcessor:
    """Emits sentences, then returns a response -- or raises."""

    def __init__(self, sentences: list[str], error: Exception | None = None) -> None:
        self.sentences = sentences
        self.error = error

    async def process(self, request, *, on_sentence=None):
        if on_sentence is not None:
            for sentence in self.sentences:
                await on_sentence(sentence)
        if self.error is not None:
            raise self.error
        return AgentTurnResponse(
            intent=TurnIntent.CONTINUE_STORY,
            response_text=" ".join(self.sentences),
            safety=SafetyDecision(),
        )


def app_for(processor) -> FastAPI:
    app = FastAPI()
    app.include_router(agent.router, prefix="/v1")
    app.dependency_overrides[get_turn_processor] = lambda: processor
    return app


async def post_stream(app: FastAPI, body: dict) -> list[dict]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/v1/agent/turn/stream", json=body) as response:
            assert response.status_code == 200
            return [json.loads(line) async for line in response.aiter_lines() if line.strip()]


async def test_emits_each_sentence_then_the_turn():
    processor = ScriptedProcessor(["Oh, a seal pup!", "What next?"])
    events = await post_stream(app_for(processor), {"transcript": "a seal pup"})

    assert [e["sentence"] for e in events if "sentence" in e] == [
        "Oh, a seal pup!",
        "What next?",
    ]
    assert events[-1]["turn"]["response_text"] == "Oh, a seal pup! What next?"
    assert sum(1 for e in events if "turn" in e) == 1


async def test_sentences_precede_the_final_turn():
    processor = ScriptedProcessor(["One.", "Two.", "Three."])
    events = await post_stream(app_for(processor), {"transcript": "hi"})

    kinds = ["sentence" if "sentence" in e else "turn" for e in events]
    assert kinds == ["sentence", "sentence", "sentence", "turn"]


@pytest.mark.parametrize(
    "error,status",
    [
        (OllamaUnavailableError("ollama down"), 503),
        (KeyError("story_missing"), 404),
        (ValueError("bad input"), 400),
        (RuntimeError("boom"), 500),
    ],
)
async def test_errors_are_reported_in_the_stream(error, status):
    """The response has already started, so errors arrive as a final event."""
    processor = ScriptedProcessor(["Partial."], error=error)
    events = await post_stream(app_for(processor), {"transcript": "hi"})

    assert events[0]["sentence"] == "Partial."
    assert events[-1]["status"] == status
    assert "error" in events[-1]


async def test_stream_terminates_even_with_no_sentences():
    processor = ScriptedProcessor([])
    events = await post_stream(app_for(processor), {"transcript": "hi"})
    assert len(events) == 1
    assert "turn" in events[0]
