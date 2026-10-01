"""The storage integration suite.

Covers `SqliteStoryRepository` against a real database file rather than a fake.
It needs no server, so it runs everywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest_asyncio

from yarnotron.core.config import Settings
from yarnotron.models.agent import TurnIntent
from yarnotron.models.media import CreateMediaAssetRequest, MediaAssetStatus, ReviewMediaAssetRequest
from yarnotron.models.interests import ChildInterest
from yarnotron.models.memory import ConversationTurn, MemoryChunk
from yarnotron.models.operations import (
    STRIKE_KEY,
    CreateSessionRequest,
    ReviewSafetyEventRequest,
    SafetyEvent,
)
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import CreateStoryRequest, StoryPatch, StoryPatchOperation
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


def embedding(*values: float) -> list[float]:
    padded = list(values) + [0.0] * (DIMENSIONS - len(values))
    return padded[:DIMENSIONS]


async def test_sessions_create_list_and_end(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Session Story"))
    session = await repository.create_session(
        CreateSessionRequest(child_id="child", story_id=story.id, device_id="mic")
    )

    listed = next(item for item in await repository.list_sessions() if item.id == session.id)
    assert listed.child_id == "child"
    assert listed.story_id == story.id
    assert listed.device_id == "mic"
    assert listed.ended_at is None

    ended_at = datetime.now(UTC)
    ended = await repository.end_session(session.id, ended_at=ended_at)
    assert ended.ended_at is not None


async def test_create_uses_readable_unique_story_ids(repository) -> None:
    first = await repository.create(CreateStoryRequest(title="The Moon Clock"))
    second = await repository.create(CreateStoryRequest(title="The Moon Clock"))

    assert first.id == "story_the_moon_clock"
    assert second.id == "story_the_moon_clock_2"


async def test_session_rejects_missing_story_id(repository) -> None:
    try:
        await repository.create_session(
            CreateSessionRequest(child_id="child", story_id="story_missing")
        )
    except KeyError:
        return
    raise AssertionError("expected KeyError for a story that does not exist")


async def test_session_can_attach_story_after_first_turn(repository) -> None:
    session = await repository.create_session(CreateSessionRequest(child_id="child"))
    story = await repository.create(CreateStoryRequest(title="Attached Later"))

    attached = await repository.attach_session_story(session.id, story.id)
    assert attached.story_id == story.id
    listed = next(item for item in await repository.list_sessions() if item.id == session.id)
    assert listed.story_id == story.id


async def test_safety_events_persist_review_and_update_session_summary(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Safety Story"))
    session = await repository.create_session(
        CreateSessionRequest(child_id="child", story_id=story.id)
    )
    event = await repository.save_safety_event(
        SafetyEvent(
            child_id="child",
            story_id=story.id,
            session_id=session.id,
            safety=SafetyDecision(
                action=SafetyAction.REDIRECT,
                severity=SafetySeverity.REVIEW,
                categories=["redirect_recommended"],
            ),
            transcript_excerpt="something to steer",
        )
    )

    listed = next(item for item in await repository.list_safety_events() if item.id == event.id)
    assert listed.safety.action == SafetyAction.REDIRECT

    reviewed = await repository.review_safety_event(
        event.id,
        ReviewSafetyEventRequest(reviewed_by="parent", review_note="looked at it"),
    )
    assert reviewed.reviewed_by == "parent"
    assert reviewed.review_note == "looked at it"
    assert reviewed.reviewed_at is not None

    # The session keeps a running count of what happened in it.
    session_after = next(
        item for item in await repository.list_sessions() if item.id == session.id
    )
    assert session_after.safety_summary.get("redirect") == 1


async def test_version_restore_round_trips_previous_story_document(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Version Story"))
    await repository.apply_patch(
        story.id,
        StoryPatch(operation=StoryPatchOperation.ADD_FACT, data={"text": "First fact."}),
    )
    after_first = await repository.get(story.id)
    assert len(after_first.facts) == 1

    await repository.apply_patch(
        story.id,
        StoryPatch(operation=StoryPatchOperation.ADD_FACT, data={"text": "Second fact."}),
    )
    assert len((await repository.get(story.id)).facts) == 2

    versions = await repository.list_versions(story.id)
    restored = await repository.restore_version(story.id, versions[0].id)
    assert len(restored.restored_story.facts) == 1
    assert restored.previous_version_id

    # And the undo is itself undoable, because restoring snapshots first.
    assert len((await repository.get(story.id)).facts) == 1


async def test_memory_search_returns_nearest_story_memory(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Memory Story"))
    other = await repository.create(CreateStoryRequest(title="Other Story"))

    await repository.save_memory_chunks(
        [
            MemoryChunk(
                story_id=story.id,
                content="The lighthouse lamp runs on whale oil.",
                embedding=embedding(1.0, 0.0),
                metadata={"kind": "fact"},
            ),
            MemoryChunk(
                story_id=story.id,
                content="Barnacle cannot fly in the rain.",
                embedding=embedding(0.0, 1.0),
            ),
            MemoryChunk(
                story_id=other.id,
                content="A different story entirely.",
                embedding=embedding(1.0, 0.0),
            ),
        ]
    )

    results = await repository.search_memory(embedding(1.0, 0.0), story_id=story.id, limit=2)
    assert results
    assert results[0].content == "The lighthouse lamp runs on whale oil."
    assert results[0].metadata == {"kind": "fact"}
    # Scoped to the story: the other story's identical vector must not appear.
    assert all(result.story_id == story.id for result in results)
    # Nearer scores higher, whichever distance metric the backend uses.
    assert results[0].score >= results[-1].score


async def test_memory_chunk_can_be_deleted(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Deletable"))
    [chunk] = await repository.save_memory_chunks(
        [MemoryChunk(story_id=story.id, content="temporary", embedding=embedding(1.0))]
    )

    assert await repository.delete_memory_chunk(chunk.id) is True
    assert await repository.delete_memory_chunk(chunk.id) is False
    assert await repository.search_memory(embedding(1.0), story_id=story.id) == []


async def test_lists_recent_conversation_turns(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Turn Story"))
    for index in range(3):
        await repository.save_conversation_turn(
            ConversationTurn(
                child_id="child",
                story_id=story.id,
                intent=TurnIntent.CONTINUE_STORY,
                transcript=f"transcript {index}",
                response_text=f"response {index}",
                safety=SafetyDecision(),
            )
        )

    turns = await repository.list_conversation_turns(limit=2)
    assert len(turns) == 2
    # Most recent first.
    assert turns[0].transcript == "transcript 2"


async def test_story_patch_persistence_records_patch_rows(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Patch Story"))
    await repository.apply_patch(
        story.id,
        StoryPatch(
            operation=StoryPatchOperation.ADD_CHARACTER,
            data={"name": "Wren"},
            rationale="The child introduced Wren.",
        ),
    )

    updated = await repository.get(story.id)
    assert [character.name for character in updated.characters] == ["Wren"]
    assert len(await repository.list_versions(story.id)) == 1


async def test_delete_story_removes_it(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Doomed"))
    assert await repository.delete(story.id) is True
    assert await repository.get(story.id) is None
    assert await repository.delete(story.id) is False


async def test_media_asset_lifecycle(repository) -> None:
    story = await repository.create(CreateStoryRequest(title="Media Story"))
    asset = await repository.create_media_asset(
        CreateMediaAssetRequest(child_id="child", story_id=story.id, prompt="a lighthouse"),
        SafetyDecision(),
    )
    assert asset.status == MediaAssetStatus.PENDING_PARENT_REVIEW

    approved = await repository.approve_media_asset(
        asset.id, ReviewMediaAssetRequest(reviewed_by="parent", parent_note="fine")
    )
    assert approved.status == MediaAssetStatus.APPROVED
    assert approved.parent_approved_by == "parent"

    generated = await repository.mark_media_asset_generated(
        asset.id,
        asset_path="/tmp/a.png",
        asset_url="/media/a.png",
        mime_type="image/png",
        generator="test",
    )
    assert generated.status == MediaAssetStatus.GENERATED
    assert [item.id for item in await repository.list_media_assets()] == [asset.id]

    failed = await repository.mark_media_asset_failed(asset.id, "generator exploded")
    assert failed.status == MediaAssetStatus.FAILED
    assert failed.failure_reason == "generator exploded"


async def test_data_survives_reopening_the_file(tmp_path) -> None:
    """The point of the whole phase: the datastore is a file that persists."""
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'persist.db'}",
        embedding_dimensions=DIMENSIONS,
    )
    first = SqliteStoryRepository(settings)
    await first.initialize()
    story = await first.create(CreateStoryRequest(title="Persisted"))
    await first.save_memory_chunks(
        [MemoryChunk(story_id=story.id, content="remembered", embedding=embedding(1.0))]
    )

    reopened = SqliteStoryRepository(settings)
    await reopened.initialize()
    assert (await reopened.get(story.id)).title == "Persisted"
    results = await reopened.search_memory(embedding(1.0), story_id=story.id)
    assert [result.content for result in results] == ["remembered"]


async def test_the_session_summary_counts_strikes_apart_from_actions(repository) -> None:
    """The processor reads the strike count back off the session next turn."""
    session = await repository.create_session(CreateSessionRequest(child_id="child_a"))
    redirect = SafetyDecision(action=SafetyAction.REDIRECT, severity=SafetySeverity.REVIEW)

    for counted in (True, True, False):
        await repository.save_safety_event(
            SafetyEvent(
                child_id="child_a",
                session_id=session.id,
                safety=redirect,
                transcript_excerpt="steal the key",
                counted_as_strike=counted,
            )
        )

    summary = (await repository.get_session(session.id)).safety_summary
    # Three redirects, but only the two the child caused are strikes.
    assert summary["redirect"] == 3
    assert summary[STRIKE_KEY] == 2


async def test_ending_a_session_is_visible_when_read_back(repository) -> None:
    session = await repository.create_session(CreateSessionRequest(child_id="child_a"))
    assert (await repository.get_session(session.id)).ended_at is None

    await repository.end_session(session.id)

    assert (await repository.get_session(session.id)).ended_at is not None


async def test_interests_belong_to_the_child_not_the_story(repository) -> None:
    """The point of the feature: an interest outlives the story it came up in."""
    await repository.save_child_interest(
        ChildInterest(child_id="child_a", name="Kerbal Space Program")
    )
    await repository.save_child_interest(ChildInterest(child_id="child_a", name="Orcas"))
    await repository.save_child_interest(ChildInterest(child_id="child_b", name="Origami"))

    names = {item.name for item in await repository.list_child_interests("child_a")}
    assert names == {"Kerbal Space Program", "Orcas"}
    assert [item.name for item in await repository.list_child_interests("child_b")] == ["Origami"]


async def test_rementioning_an_interest_updates_it_rather_than_duplicating(repository) -> None:
    first = await repository.save_child_interest(
        ChildInterest(child_id="child_a", name="Kerbal Space Program")
    )
    # The same interest, said differently, and now with a description.
    await repository.save_child_interest(
        ChildInterest(
            child_id="child_a",
            name="kerbal  space  program",
            summary="A game about building wobbly rockets.",
        )
    )

    interests = await repository.list_child_interests("child_a")
    assert len(interests) == 1
    assert interests[0].summary == "A game about building wobbly rockets."
    # The row keeps its identity, so the background lookup cannot write to a
    # record that moved underneath it.
    assert interests[0].id == first.id


async def test_interests_survive_reopening_the_file(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'interests.db'}",
        embedding_dimensions=DIMENSIONS,
    )
    first = SqliteStoryRepository(settings)
    await first.initialize()
    await first.save_child_interest(
        ChildInterest(child_id="child_a", name="Kerbal Space Program")
    )

    reopened = SqliteStoryRepository(settings)
    await reopened.initialize()
    found = await reopened.get_child_interest("child_a", "KERBAL SPACE PROGRAM")
    assert found is not None and found.name == "Kerbal Space Program"
