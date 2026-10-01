"""GET /v1/status: one cheap call for a client that polls (the Home Assistant integration).

Counting stories, sessions, safety events and media by listing them would ship
every full story document over the network each poll. This asks SQLite to count.
"""

from __future__ import annotations

import httpx
import pytest_asyncio
from fastapi import FastAPI

from yarnotron.api.routes import status as status_routes
from yarnotron.core.config import Settings, get_settings
from yarnotron.models.media import CreateMediaAssetRequest
from yarnotron.models.operations import (
    STRIKE_KEY,
    CreateSessionRequest,
    ReviewSafetyEventRequest,
    SafetyEvent,
)
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import CreateStoryRequest
from yarnotron.services.ollama import OllamaModel, OllamaUnavailableError, get_ollama_client
from yarnotron.services.sqlite_story_repository import SqliteStoryRepository

DIMENSIONS = 8


@pytest_asyncio.fixture
async def repository(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'yarnotron.db'}",
        embedding_dimensions=DIMENSIONS,
    )
    store = SqliteStoryRepository(settings)
    await store.initialize()
    return store


def caution(child_id: str, session_id: str | None = None) -> SafetyEvent:
    return SafetyEvent(
        child_id=child_id,
        session_id=session_id,
        safety=SafetyDecision(
            action=SafetyAction.REDIRECT,
            severity=SafetySeverity.REVIEW,
            categories=["gore"],
        ),
        transcript_excerpt="x",
        counted_as_strike=True,
    )


class FakeOllama:
    def __init__(self, names: list[str] | None = None, down: bool = False) -> None:
        self.names = names or []
        self.down = down

    async def list_models(self):
        if self.down:
            raise OllamaUnavailableError("connection refused")
        return [OllamaModel(name=name) for name in self.names]


def make_app(repository, ollama, **settings_overrides) -> FastAPI:
    settings = Settings(
        default_model="chat:1",
        safety_model="safe:1",
        embedding_model="embed:1",
        safety_strike_limit=4,
        **settings_overrides,
    )
    app = FastAPI(version="9.9.9")
    app.state.story_repository = repository
    app.include_router(status_routes.router, prefix="/v1")
    app.dependency_overrides[get_ollama_client] = lambda: ollama
    app.dependency_overrides[get_settings] = lambda: settings
    return app


async def get_status(app: FastAPI, **params) -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/status", params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def test_summary_counts_belong_to_the_child_asked_about(repository) -> None:
    await repository.create(CreateStoryRequest(title="One"))
    await repository.create(CreateStoryRequest(title="Two"))
    await repository.create_session(CreateSessionRequest(child_id="mia"))
    await repository.create_session(CreateSessionRequest(child_id="mia"))
    await repository.create_session(CreateSessionRequest(child_id="sam"))
    await repository.save_safety_event(caution("mia"))
    await repository.save_safety_event(caution("sam"))
    await repository.save_safety_event(caution("sam"))
    await repository.create_media_asset(
        CreateMediaAssetRequest(child_id="mia", prompt="a dragon"),
        safety=SafetyDecision(),
    )

    mia = await repository.status_summary("mia")
    sam = await repository.status_summary("sam")

    assert mia.stories == sam.stories == 2  # stories are the household's, not a child's
    assert (mia.open_sessions, sam.open_sessions) == (2, 1)
    assert (mia.unreviewed_safety_events, sam.unreviewed_safety_events) == (1, 2)
    assert (mia.pending_media, sam.pending_media) == (1, 0)


async def test_reviewed_events_and_ended_sessions_stop_counting(repository) -> None:
    session = await repository.create_session(CreateSessionRequest(child_id="mia"))
    event = await repository.save_safety_event(caution("mia", session.id))
    assert (await repository.status_summary("mia")).unreviewed_safety_events == 1

    await repository.review_safety_event(event.id, ReviewSafetyEventRequest())
    await repository.end_session(session.id)

    summary = await repository.status_summary("mia")
    assert summary.unreviewed_safety_events == 0
    assert summary.open_sessions == 0


async def test_latest_session_carries_ended_state_and_strikes(repository) -> None:
    older = await repository.create_session(CreateSessionRequest(child_id="mia"))
    await repository.end_session(older.id)
    newest = await repository.create_session(CreateSessionRequest(child_id="mia"))
    await repository.save_safety_event(caution("mia", newest.id))
    await repository.save_safety_event(caution("mia", newest.id))

    summary = await repository.status_summary("mia")

    assert summary.latest_session_id == newest.id
    assert summary.latest_session_ended is False
    assert summary.latest_session_strikes == 2 == (
        (await repository.get_session(newest.id)).safety_summary[STRIKE_KEY]
    )

    await repository.end_session(newest.id)
    assert (await repository.status_summary("mia")).latest_session_ended is True


async def test_a_child_with_no_history_is_all_zeros(repository) -> None:
    summary = await repository.status_summary("nobody")

    assert summary.latest_session_id is None
    assert summary.latest_session_ended is False
    assert summary.latest_session_strikes == 0
    assert summary.open_sessions == summary.unreviewed_safety_events == 0


async def test_route_reports_models_limits_and_auth(repository) -> None:
    app = make_app(
        repository,
        FakeOllama(["chat:1", "safe:1", "embed:1", "other:1"]),
        api_token="tok",
    )

    body = await get_status(app, child_id="mia")

    assert body["app"] == "Yarnotron"
    assert body["version"] == "9.9.9"
    assert body["auth_required"] is True
    assert body["strike_limit"] == 4
    assert body["models"] == {
        "ollama_reachable": True,
        "chat_model": "chat:1",
        "safety_model": "safe:1",
        "embedding_model": "embed:1",
        "missing": [],
    }
    assert body["child"]["child_id"] == "mia"


async def test_route_names_the_models_that_are_not_installed(repository) -> None:
    app = make_app(repository, FakeOllama(["chat:1"]))

    body = await get_status(app)

    assert body["models"]["ollama_reachable"] is True
    assert body["models"]["missing"] == ["safe:1", "embed:1"]
    assert body["auth_required"] is False


async def test_route_still_answers_when_ollama_is_down(repository) -> None:
    # "Yarnotron is up but the model server is not" is exactly what a client
    # needs to be told, so this must be a 200 with a flag, not a 503.
    app = make_app(repository, FakeOllama(down=True))

    body = await get_status(app)

    assert body["models"]["ollama_reachable"] is False
    assert body["models"]["missing"] == []  # unknown, not "all missing"


async def test_route_defaults_to_the_local_child(repository) -> None:
    app = make_app(repository, FakeOllama())

    assert (await get_status(app))["child"]["child_id"] == "local_child"
