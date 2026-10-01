"""A parent overriding a block, and the child getting the turn back.

A block stops a turn outright and writes nothing. When it was wrong -- a story
about a dog read as sexual content -- the child has simply lost the idea. A
parent releasing it re-runs the turn, and the child is told what landed the next
time they open that story, since they were not there when it did.
"""

from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.operations import CreateSessionRequest, ReleaseSafetyEventRequest
from yarnotron.services.memory import StoryMemoryBuilder

from tests.test_turn_processor import (
    FakeEmbedder,
    FakeOllamaClient,
    FakeSafetyReviewer,
    FakeStoryRepository,
)
from tests.test_safety_strikes import BlockingInputReviewer

LINE = "She deflates the toys sometimes when she bites them."


def build(repository, safety, ollama=None) -> TurnProcessor:
    return TurnProcessor(
        ollama=ollama or FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=safety,
    )


async def blocked_turn(repository) -> tuple[str, str]:
    """Run the turn that gets wrongly blocked. Returns (session_id, event_id)."""
    session = await repository.create_session(CreateSessionRequest(story_id="story_test"))
    processor = build(repository, BlockingInputReviewer())
    await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session.id, transcript=LINE)
    )
    return session.id, repository.safety_events[0].id


async def test_a_blocked_turn_writes_nothing() -> None:
    repository = FakeStoryRepository()
    await blocked_turn(repository)

    assert repository.turns[0].applied_story_patches == []


async def test_releasing_re_runs_the_turn_and_lands_it_in_the_story() -> None:
    repository = FakeStoryRepository()
    session_id, event_id = await blocked_turn(repository)

    await repository.release_safety_event(event_id, ReleaseSafetyEventRequest())
    # The reviewer would block it again; the release is what gets it through.
    processor = build(repository, BlockingInputReviewer())
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript=LINE),
        parent_released=True,
    )

    assert len(response.applied_story_patches) == 1
    assert response.safety.categories == ["parent_released"]


async def test_the_child_is_told_what_landed_next_time_they_open_the_story() -> None:
    repository = FakeStoryRepository()
    session_id, event_id = await blocked_turn(repository)
    await repository.release_safety_event(event_id, ReleaseSafetyEventRequest())
    processor = build(repository, BlockingInputReviewer())
    await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript=LINE),
        parent_released=True,
    )

    # The child comes back. A different reviewer, because this line is fine.
    later = build(repository, FakeSafetyReviewer())
    response = await later.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=session_id,
            transcript="What happens next?",
        )
    )

    assert "where we left off" in response.response_text
    assert LINE in response.response_text


async def test_the_child_is_only_told_once() -> None:
    repository = FakeStoryRepository()
    session_id, event_id = await blocked_turn(repository)
    await repository.release_safety_event(event_id, ReleaseSafetyEventRequest())
    processor = build(repository, FakeSafetyReviewer())

    first = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript="Go on.")
    )
    second = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript="And then?")
    )

    assert "where we left off" in first.response_text
    assert "where we left off" not in second.response_text


async def test_the_re_run_itself_does_not_get_the_recap() -> None:
    """Nobody is listening to the re-run, so the recap is saved for the child."""
    repository = FakeStoryRepository()
    session_id, event_id = await blocked_turn(repository)
    await repository.release_safety_event(event_id, ReleaseSafetyEventRequest())
    processor = build(repository, BlockingInputReviewer())

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript=LINE),
        parent_released=True,
    )

    assert "where we left off" not in response.response_text
    assert repository.safety_events[0].recap_delivered_at is None


async def test_a_released_line_stays_allowed_for_the_rest_of_the_session() -> None:
    """The child can revisit the idea without it being stopped again."""
    repository = FakeStoryRepository()
    session_id, event_id = await blocked_turn(repository)
    await repository.release_safety_event(event_id, ReleaseSafetyEventRequest())
    processor = build(repository, BlockingInputReviewer())

    # Said again by the child, no parent_released flag, reviewer still hostile.
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript=LINE)
    )

    assert len(response.applied_story_patches) == 1
    assert "safety alert" not in response.response_text


async def test_the_release_does_not_leak_into_another_session() -> None:
    repository = FakeStoryRepository()
    _, event_id = await blocked_turn(repository)
    await repository.release_safety_event(event_id, ReleaseSafetyEventRequest())

    other = await repository.create_session(CreateSessionRequest(story_id="story_test"))
    processor = build(repository, BlockingInputReviewer())
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=other.id, transcript=LINE)
    )

    assert "safety alert" in response.response_text
    assert response.applied_story_patches == []


async def test_releasing_records_who_did_it() -> None:
    repository = FakeStoryRepository()
    _, event_id = await blocked_turn(repository)

    released = await repository.release_safety_event(
        event_id,
        ReleaseSafetyEventRequest(released_by="a_parent", review_note="It is a dog."),
    )

    assert released.released_at is not None
    assert released.released_by == "a_parent"
    assert released.review_note == "It is a dog."
    # Releasing settles the review too; it should not still read as outstanding.
    assert released.reviewed_at is not None
