"""The release endpoint's guards. Only a lost turn can be handed back."""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from yarnotron.agent.turn_processor import get_turn_processor
from yarnotron.api.routes import safety as safety_routes
from yarnotron.models.operations import SafetyEvent
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.services.story_store import get_story_repository

from tests.test_turn_processor import FakeStoryRepository


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def process(self, request, *, on_sentence=None, parent_released=False):
        self.calls.append((request.transcript, parent_released))
        return None


def app_for(repository, processor) -> FastAPI:
    app = FastAPI()
    app.include_router(safety_routes.router, prefix="/v1")
    app.dependency_overrides[get_story_repository] = lambda: repository
    app.dependency_overrides[get_turn_processor] = lambda: processor
    return app


async def post_release(app: FastAPI, event_id: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(f"/v1/safety-events/{event_id}/release", json={})


def blocked_event(**overrides) -> SafetyEvent:
    defaults = {
        "child_id": "local_child",
        "story_id": "story_test",
        "session_id": "session_test",
        "safety": SafetyDecision(
            action=SafetyAction.BLOCK,
            severity=SafetySeverity.ALERT,
            categories=["sexual content"],
        ),
        "transcript_excerpt": "She deflates the toys sometimes when she bites them.",
    }
    return SafetyEvent(**{**defaults, **overrides})


async def test_releasing_records_it_and_re_runs_the_turn() -> None:
    repository = FakeStoryRepository()
    event = blocked_event()
    repository.safety_events.append(event)
    processor = RecordingProcessor()

    response = await post_release(app_for(repository, processor), event.id)

    assert response.status_code == 200
    assert response.json()["released_at"] is not None
    assert processor.calls == [
        ("She deflates the toys sometimes when she bites them.", True)
    ]


async def test_the_release_is_recorded_before_the_re_run() -> None:
    """Otherwise the re-run's own safety review would block it a second time."""
    repository = FakeStoryRepository()
    event = blocked_event()
    repository.safety_events.append(event)
    seen: list[bool] = []

    class CheckingProcessor(RecordingProcessor):
        async def process(self, request, *, on_sentence=None, parent_released=False):
            seen.append(repository.safety_events[0].released_at is not None)
            return None

    await post_release(app_for(repository, CheckingProcessor()), event.id)

    assert seen == [True]


async def test_a_caution_has_nothing_to_release() -> None:
    repository = FakeStoryRepository()
    event = blocked_event(
        safety=SafetyDecision(action=SafetyAction.REDIRECT, severity=SafetySeverity.REVIEW)
    )
    repository.safety_events.append(event)
    processor = RecordingProcessor()

    response = await post_release(app_for(repository, processor), event.id)

    assert response.status_code == 409
    assert processor.calls == []


async def test_a_session_ended_refusal_has_nothing_to_release() -> None:
    repository = FakeStoryRepository()
    event = blocked_event(
        safety=SafetyDecision(
            action=SafetyAction.BLOCK,
            severity=SafetySeverity.REVIEW,
            categories=["session_ended"],
        )
    )
    repository.safety_events.append(event)
    processor = RecordingProcessor()

    response = await post_release(app_for(repository, processor), event.id)

    assert response.status_code == 409
    assert processor.calls == []


async def test_a_block_cannot_be_released_twice() -> None:
    repository = FakeStoryRepository()
    event = blocked_event()
    repository.safety_events.append(event)
    processor = RecordingProcessor()
    app = app_for(repository, processor)

    assert (await post_release(app, event.id)).status_code == 200
    assert (await post_release(app, event.id)).status_code == 409
    assert len(processor.calls) == 1


async def test_a_block_with_no_story_cannot_be_released() -> None:
    repository = FakeStoryRepository()
    event = blocked_event(story_id=None)
    repository.safety_events.append(event)
    processor = RecordingProcessor()

    response = await post_release(app_for(repository, processor), event.id)

    assert response.status_code == 409
    assert processor.calls == []


async def test_an_unknown_event_is_a_404() -> None:
    response = await post_release(
        app_for(FakeStoryRepository(), RecordingProcessor()), "safety_nope"
    )

    assert response.status_code == 404
