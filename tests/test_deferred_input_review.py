"""The model's view of what the child said, moved off the turn.

It used to sit in front of the child at ~370 ms. It now runs while Yarnotron is
speaking, when the GPU is idle anyway. What it can no longer do is stop the
turn; what it must still do is raise the parent alert and spend the strike.
"""

import asyncio

from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.operations import STRIKE_KEY, CreateSessionRequest
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.services.memory import StoryMemoryBuilder

from tests.test_turn_processor import (
    FakeEmbedder,
    FakeOllamaClient,
    FakeSafetyReviewer,
    FakeStoryRepository,
)


class LateReviewer(FakeSafetyReviewer):
    """Keywords find nothing; the model finds something, afterwards."""

    def __init__(self, action: SafetyAction, severity=SafetySeverity.ALERT) -> None:
        self.action = action
        self.severity = severity
        self.reviewed: list[str] = []

    def classify_transcript(self, transcript: str) -> SafetyDecision:
        return SafetyDecision()

    async def review_transcript(self, transcript, story=None, intent=None) -> SafetyDecision:
        self.reviewed.append(transcript)
        return SafetyDecision(
            action=self.action, severity=self.severity, categories=["late"]
        )


def build(repository, safety) -> TurnProcessor:
    return TurnProcessor(
        ollama=FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=safety,
    )


async def drain(processor: TurnProcessor) -> None:
    await asyncio.gather(*processor._input_review_tasks)


async def test_the_turn_is_not_held_up_by_the_model_review() -> None:
    repository = FakeStoryRepository()
    reviewer = LateReviewer(SafetyAction.BLOCK)
    processor = build(repository, reviewer)

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="It is a seal pup.")
    )

    # The reply landed on the child's story as normal.
    assert len(response.applied_story_patches) == 1
    assert "safety alert" not in response.response_text


async def test_a_late_block_still_alerts_the_parent() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, LateReviewer(SafetyAction.BLOCK))
    session = await repository.create_session(CreateSessionRequest(story_id="story_test"))

    await processor.process(
        AgentTurnRequest(
            story_id="story_test", session_id=session.id, transcript="It is a seal pup."
        )
    )
    await drain(processor)

    event = repository.safety_events[-1]
    assert event.safety.action == SafetyAction.BLOCK
    assert event.parent_alert_sent is True
    assert event.transcript_excerpt == "It is a seal pup."


async def test_a_late_redirect_still_spends_a_strike() -> None:
    repository = FakeStoryRepository()
    processor = build(
        repository, LateReviewer(SafetyAction.REDIRECT, SafetySeverity.REVIEW)
    )
    session = await repository.create_session(CreateSessionRequest(story_id="story_test"))

    await processor.process(
        AgentTurnRequest(
            story_id="story_test", session_id=session.id, transcript="steal the key"
        )
    )
    await drain(processor)

    assert repository.safety_events[-1].counted_as_strike is True
    live = await repository.get_session(session.id)
    assert live.safety_summary.get(STRIKE_KEY, 0) == 1


async def test_the_event_is_tied_to_the_turn_it_came_from() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, LateReviewer(SafetyAction.BLOCK))

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="It is a seal pup.")
    )
    await drain(processor)

    assert repository.safety_events[-1].conversation_turn_id == response.turn_id


async def test_an_allowed_turn_records_nothing() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, FakeSafetyReviewer())

    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="It is a seal pup.")
    )
    await drain(processor)

    assert repository.safety_events == []


async def test_a_keyword_stop_is_not_reviewed_again() -> None:
    """The guard already decided; a second opinion would only duplicate it."""

    class KeywordBlocks(FakeSafetyReviewer):
        def classify_transcript(self, transcript: str) -> SafetyDecision:
            return SafetyDecision(
                action=SafetyAction.ALERT_PARENT, severity=SafetySeverity.ALERT
            )

        async def review_transcript(self, transcript, story=None, intent=None):
            raise AssertionError("should not run for a turn the keywords stopped")

    repository = FakeStoryRepository()
    processor = build(repository, KeywordBlocks())

    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="anything")
    )
    await drain(processor)

    assert len(repository.safety_events) == 1


async def test_a_failing_review_never_breaks_the_turn() -> None:
    class ExplodingReviewer(FakeSafetyReviewer):
        def classify_transcript(self, transcript: str) -> SafetyDecision:
            return SafetyDecision()

        async def review_transcript(self, transcript, story=None, intent=None):
            raise RuntimeError("model gone")

    repository = FakeStoryRepository()
    processor = build(repository, ExplodingReviewer())

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="It is a seal pup.")
    )
    await drain(processor)

    assert len(response.applied_story_patches) == 1
    assert repository.safety_events == []
