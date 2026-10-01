"""A milder safety flag cautions the child instead of ending the turn.

Being stopped mid-sentence teaches a child not to talk, not what is safe. The
parent still gets every event; what changes is that the story keeps moving --
until a session has spent its strikes, which is where it does stop.
"""

import pytest

from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.operations import STRIKE_KEY, CreateSessionRequest
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import StoryPatch
from yarnotron.services.memory import StoryMemoryBuilder

from tests.test_turn_processor import (
    FakeEmbedder,
    FakeOllamaClient,
    FakeSafetyReviewer,
    FakeStoryRepository,
)
from tests.test_turn_processor_stream import StreamingOllamaClient

CAUTION = "Let's steer that in a safer direction."


class RedirectingInputReviewer(FakeSafetyReviewer):
    """Flags what the child said, but is happy with what the assistant writes."""

    def classify_transcript(self, transcript: str) -> SafetyDecision:
        return SafetyDecision(
            action=SafetyAction.REDIRECT,
            severity=SafetySeverity.REVIEW,
            categories=["redirect_recommended"],
            child_message=CAUTION,
        )


class AlertingInputReviewer(FakeSafetyReviewer):
    def classify_transcript(self, transcript: str) -> SafetyDecision:
        return SafetyDecision(
            action=SafetyAction.ALERT_PARENT,
            severity=SafetySeverity.ALERT,
            categories=["child_safety_review"],
            child_message="I'm going to save a note for your parent.",
        )


class RedirectingOutputReviewer(FakeSafetyReviewer):
    """The assistant wrote something unsafe. Not the child's doing."""

    async def review_assistant_output(
        self,
        transcript: str,
        response_text: str,
        story_patches: list[StoryPatch],
    ) -> SafetyDecision:
        return SafetyDecision(
            action=SafetyAction.REDIRECT,
            severity=SafetySeverity.REVIEW,
            categories=["assistant_output_review"],
        )


def build(repository, safety, *, strike_limit=5, ollama=None):
    return TurnProcessor(
        ollama=ollama or FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=safety,
        strike_limit=strike_limit,
    )


async def session_for(repository) -> str:
    session = await repository.create_session(CreateSessionRequest(story_id="story_test"))
    return session.id


async def test_a_cautioned_turn_still_moves_the_story_on() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer())

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="Wren wants to steal the lighthouse key.",
        )
    )

    assert response.response_text.startswith(CAUTION)
    # The story advanced anyway -- this is the whole point of the change.
    assert len(response.applied_story_patches) == 1
    assert response.updated_story is not None


async def test_the_caution_is_recorded_for_the_parent() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer())

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="Wren wants to steal the lighthouse key.",
        )
    )

    assert response.safety.action == SafetyAction.REDIRECT
    assert len(repository.safety_events) == 1
    assert repository.safety_events[0].counted_as_strike is True


async def test_the_caution_is_spoken_first_when_streaming() -> None:
    repository = FakeStoryRepository()
    processor = build(
        repository,
        RedirectingInputReviewer(),
        ollama=StreamingOllamaClient(),
    )
    spoken: list[str] = []

    async def on_sentence(sentence: str) -> None:
        spoken.append(sentence)

    await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="Wren wants to steal the lighthouse key.",
        ),
        on_sentence=on_sentence,
    )

    # The voice loop never re-speaks the final text once sentences streamed, so
    # the caution has to lead rather than be appended.
    assert spoken and CAUTION in spoken[0]


async def test_high_severity_still_stops_the_turn_immediately() -> None:
    """The whole reason the tiers are kept apart."""
    repository = FakeStoryRepository()
    processor = build(repository, AlertingInputReviewer())

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="something that trips the alert tier",
        )
    )

    assert response.safety.action == SafetyAction.ALERT_PARENT
    assert response.applied_story_patches == []
    assert response.updated_story is None
    # No strike: this tier does not get four more chances.
    assert repository.safety_events[0].counted_as_strike is False


class BlockingInputReviewer(FakeSafetyReviewer):
    def classify_transcript(self, transcript: str) -> SafetyDecision:
        return SafetyDecision(
            action=SafetyAction.BLOCK,
            severity=SafetySeverity.ALERT,
            categories=["sexual content"],
            # The wording a model actually produced when it misread a story
            # about a dog. It says nothing about safety or about a grown-up.
            child_message="I can't help with that story. Let's try a different adventure!",
            parent_note="Model reasoning, for the dashboard.",
        )


async def test_a_block_raises_a_parent_alert() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, BlockingInputReviewer())

    await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="She deflates the toys sometimes when she bites them.",
        )
    )

    event = repository.safety_events[0]
    assert event.parent_alert_sent is True
    # The reviewer's own reasoning is still kept for the parent to read.
    assert event.safety.parent_note == "Model reasoning, for the dashboard."


async def test_a_blocked_child_is_told_it_was_a_safety_stop() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, BlockingInputReviewer())

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="She deflates the toys sometimes when she bites them.",
        )
    )

    assert "safety alert" in response.response_text
    assert "grown-up" in response.response_text
    # Fixed wording, not whatever the reviewer improvised.
    assert "different adventure" not in response.response_text


async def test_an_alert_tier_stop_sends_the_child_to_an_adult() -> None:
    """Serious content should not end on 'tell me a different idea'."""
    repository = FakeStoryRepository()
    processor = build(repository, AlertingInputReviewer())

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=await session_for(repository),
            transcript="something that trips the alert tier",
        )
    )

    assert "talk with them" in response.response_text
    assert repository.safety_events[0].parent_alert_sent is True


async def test_being_turned_away_after_a_cut_off_is_not_a_parent_alert() -> None:
    """Otherwise every word after a cut-off would alert a parent again."""
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=1)
    session_id = await session_for(repository)

    await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=session_id, transcript="steal the key")
    )
    for _ in range(3):
        await processor.process(
            AgentTurnRequest(story_id="story_test", session_id=session_id, transcript="hello?")
        )

    refusals = [e for e in repository.safety_events if "session_ended" in e.safety.categories]
    assert len(refusals) == 3
    assert not any(e.parent_alert_sent for e in refusals)


async def test_the_fifth_strike_ends_the_session() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=5)
    session_id = await session_for(repository)

    responses = []
    for _ in range(5):
        responses.append(
            await processor.process(
                AgentTurnRequest(
                    story_id="story_test",
                    session_id=session_id,
                    transcript="Wren wants to steal the lighthouse key.",
                )
            )
        )

    # The first four carried on with a caution.
    for response in responses[:4]:
        assert response.response_text.startswith(CAUTION)
        assert len(response.applied_story_patches) == 1

    final = responses[4]
    assert "take a break" in final.response_text
    assert final.applied_story_patches == []
    assert "strike_limit_reached" in final.safety.categories
    assert (await repository.get_session(session_id)).ended_at is not None


async def test_a_turn_after_the_session_ended_is_turned_away_without_a_model_call() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=2, ollama=ollama)
    session_id = await session_for(repository)

    for _ in range(2):
        await processor.process(
            AgentTurnRequest(
                story_id="story_test",
                session_id=session_id,
                transcript="Wren wants to steal the lighthouse key.",
            )
        )
    calls_before = ollama.calls

    after = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=session_id,
            transcript="Can we keep going?",
        )
    )

    assert "grown-up" in after.response_text
    assert ollama.calls == calls_before


async def test_a_redirect_caused_by_the_assistant_is_not_the_childs_strike() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingOutputReviewer())
    session_id = await session_for(repository)

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            session_id=session_id,
            transcript="The seal pup follows Wren up the stairs.",
        )
    )

    assert response.safety.action == SafetyAction.REDIRECT
    assert repository.safety_events[0].counted_as_strike is False
    assert (await repository.get_session(session_id)).safety_summary.get(STRIKE_KEY, 0) == 0


@pytest.mark.parametrize("limit", [1, 3])
async def test_the_strike_limit_is_configurable(limit: int) -> None:
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=limit)
    session_id = await session_for(repository)

    for _ in range(limit):
        response = await processor.process(
            AgentTurnRequest(
                story_id="story_test",
                session_id=session_id,
                transcript="Wren wants to steal the lighthouse key.",
            )
        )

    assert "take a break" in response.response_text
    assert (await repository.get_session(session_id)).ended_at is not None


async def test_strikes_accumulate_even_when_the_caller_sends_no_session() -> None:
    """The gap this closes: no session id used to mean no strike ever landed."""
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=3)

    for _ in range(3):
        response = await processor.process(
            AgentTurnRequest(story_id="story_test", transcript="steal the key")
        )

    assert "take a break" in response.response_text
    assert len(repository.sessions) == 1
    assert list(repository.sessions.values())[0].ended_at is not None


async def test_a_sessionless_turn_is_told_which_session_it_landed_in() -> None:
    repository = FakeStoryRepository()
    processor = build(repository, FakeSafetyReviewer())

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="It is a seal pup.")
    )

    assert response.session_id is not None
    assert response.session_id in repository.sessions


async def test_a_sessionless_turn_cannot_walk_away_from_an_ended_session() -> None:
    """Omitting the field must not be a way to start over."""
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=1)

    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="steal the key")
    )
    after = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="can we keep going?")
    )

    assert "grown-up" in after.response_text
    # Still one session: the turn attached to the ended one instead of minting
    # a fresh count for itself.
    assert len(repository.sessions) == 1


async def test_a_grown_up_starting_a_session_clears_the_way_again() -> None:
    """The documented escape hatch, and the only one."""
    repository = FakeStoryRepository()
    # Two, so the new session has a caution left to spend and the reset is
    # visible as something other than an immediate second cut-off.
    processor = build(repository, RedirectingInputReviewer(), strike_limit=2)

    for _ in range(2):
        await processor.process(
            AgentTurnRequest(story_id="story_test", transcript="steal the key")
        )
    assert "grown-up" in (
        await processor.process(
            AgentTurnRequest(story_id="story_test", transcript="hello?")
        )
    ).response_text

    await repository.create_session(CreateSessionRequest(story_id="story_test"))
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="steal the key")
    )

    assert response.response_text.startswith(CAUTION)


async def test_strikes_are_scoped_to_one_session() -> None:
    """Ending a session is a cooling-off, not a lockout: a new one starts clean."""
    repository = FakeStoryRepository()
    processor = build(repository, RedirectingInputReviewer(), strike_limit=2)

    first = await session_for(repository)
    for _ in range(2):
        await processor.process(
            AgentTurnRequest(story_id="story_test", session_id=first, transcript="steal the key")
        )
    assert (await repository.get_session(first)).ended_at is not None

    second = await session_for(repository)
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", session_id=second, transcript="steal the key")
    )

    assert response.response_text.startswith(CAUTION)
    assert len(response.applied_story_patches) == 1
    assert (await repository.get_session(second)).ended_at is None
