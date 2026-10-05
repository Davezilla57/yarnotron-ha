"""Graceful shutdown: drain running work, end sessions, report what was saved."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from yarnotron.api.routes import sessions as session_routes
from yarnotron.api.routes import status as status_routes
from yarnotron.api.routes import system as system_routes
from yarnotron.core.config import Settings, get_settings
from yarnotron.models.operations import CreateSessionRequest
from yarnotron.models.story import CreateStoryRequest
from yarnotron.services.ollama import get_ollama_client
from yarnotron.services.shutdown import ShutdownCoordinator, get_shutdown, reset_shutdown
from yarnotron.services.sqlite_story_repository import SqliteStoryRepository


@pytest.fixture(autouse=True)
def fresh_coordinator():
    return reset_shutdown()


@pytest_asyncio.fixture
async def repository(tmp_path):
    store = SqliteStoryRepository(
        Settings(database_url=f"sqlite:///{tmp_path / 'y.db'}", embedding_dimensions=8)
    )
    await store.initialize()
    return store


class FakeOllama:
    async def list_models(self):
        return []


def make_app(repository) -> FastAPI:
    app = FastAPI()
    app.state.story_repository = repository
    for module in (system_routes, session_routes, status_routes):
        app.include_router(module.router, prefix="/v1")
    app.dependency_overrides[get_ollama_client] = lambda: FakeOllama()
    app.dependency_overrides[get_settings] = lambda: Settings()
    return app


def client_for(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_drain_waits_for_background_work() -> None:
    coordinator = ShutdownCoordinator()

    async def review() -> None:
        await asyncio.sleep(0.2)

    coordinator.track_task(asyncio.create_task(review()))
    report = await coordinator.drain(timeout=2)

    assert report.drained
    assert report.background_tasks_finished == 1
    assert report.background_tasks_cancelled == 0


async def test_drain_cancels_work_that_outlasts_the_timeout() -> None:
    coordinator = ShutdownCoordinator()

    async def stuck() -> None:
        await asyncio.sleep(60)

    task = asyncio.create_task(stuck())
    coordinator.track_task(task)
    report = await coordinator.drain(timeout=0.2)

    assert not report.drained
    assert report.background_tasks_cancelled == 1
    assert task.cancelled()


async def test_drain_reports_a_turn_that_never_finishes() -> None:
    coordinator = ShutdownCoordinator()
    with coordinator.turn():
        report = await coordinator.drain(timeout=0.2)
    assert not report.drained
    assert report.turns_still_running == 1


async def test_prepare_shutdown_ends_sessions_and_reports_the_story(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="The Brave Fox"))
    session = await repository.create_session(
        CreateSessionRequest(child_id="mia", story_id=story.id)
    )
    app = make_app(repository)

    async with client_for(app) as client:
        response = await client.post("/v1/system/prepare-shutdown", params={"timeout": 1})
        body = response.json()
        status = (await client.get("/v1/status")).json()

    assert response.status_code == 200, response.text
    assert body["drained"] is True
    assert body["sessions_ended"] == [session.id]
    assert body["story"]["title"] == "The Brave Fox"
    assert body["checkpoint"] == "ok"
    assert status["shutting_down"] is True
    assert all(s.ended_at for s in await repository.list_sessions())


async def test_new_sessions_are_refused_while_draining_and_cancel_resumes(repository) -> None:
    app = make_app(repository)
    async with client_for(app) as client:
        await client.post("/v1/system/prepare-shutdown", params={"timeout": 1})
        refused = await client.post("/v1/sessions", json={"child_id": "mia"})
        resumed = await client.post("/v1/system/cancel-shutdown")
        allowed = await client.post("/v1/sessions", json={"child_id": "mia"})

    assert refused.status_code == 503
    assert "shutting down" in refused.json()["detail"]
    assert resumed.json() == {"draining": False}
    assert allowed.status_code == 201


async def test_prepare_shutdown_is_safe_twice_and_with_nothing_open(repository) -> None:
    app = make_app(repository)
    async with client_for(app) as client:
        first = await client.post("/v1/system/prepare-shutdown", params={"timeout": 1})
        second = await client.post("/v1/system/prepare-shutdown", params={"timeout": 1})
    assert first.status_code == second.status_code == 200
    assert second.json()["story"] is None


async def test_notes_explain_a_partial_drain(repository) -> None:
    app = make_app(repository)
    with get_shutdown().turn():
        async with client_for(app) as client:
            body = (
                await client.post("/v1/system/prepare-shutdown", params={"timeout": 0.2})
            ).json()
    assert body["drained"] is False
    assert any("still running" in note for note in body["notes"])
