"""Interests a child brings in from outside the story they are writing."""

import asyncio

from yarnotron.agent.prompts import build_story_turn_prompt
from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest, TurnIntent
from yarnotron.models.interests import ChildInterest
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import StoryPatch
from yarnotron.services.interests import InterestEnricher
from yarnotron.services.memory import StoryMemoryBuilder
from tests.test_turn_processor import (
    FakeEmbedder,
    FakeOllamaClient,
    FakeSafetyReviewer,
    FakeStoryRepository,
)
from tests.test_turn_processor_stream import StreamingOllamaClient


def _draft(
    *,
    response_text: str = "A wobbling rocket is a great idea.",
    interest: dict[str, object] | None = None,
    patches: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "response_text": response_text,
        "story_patches": patches
        if patches is not None
        else [
            {
                "operation": "add_fact",
                "data": {"text": "The rocket wobbles on take-off."},
                "child_confirmation_required": False,
                "rationale": "The child added this to the story.",
            }
        ],
        "tutor_notes": [],
        "interest_candidate": interest,
    }


def _processor(
    repository: FakeStoryRepository,
    ollama: FakeOllamaClient,
    *,
    enricher: InterestEnricher | None = None,
    safety: FakeSafetyReviewer | None = None,
) -> TurnProcessor:
    return TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=safety or FakeSafetyReviewer(),
        interest_enricher=enricher,
    )


async def test_a_clear_outside_interest_is_recorded() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient(
        _draft(interest={"name": "Kerbal Space Program", "ambiguous": False})
    )

    await _processor(repository, ollama).process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="The rocket should wobble like in Kerbal Space Program.",
        )
    )

    assert [item.name for item in repository.interests] == ["Kerbal Space Program"]
    assert repository.interests[0].child_id == "local_child"
    # The description is filled in by the background lookup, not on this turn.
    assert repository.interests[0].summary is None


async def test_a_clear_interest_does_not_hold_back_story_patches() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient(
        _draft(interest={"name": "Kerbal Space Program", "ambiguous": False})
    )

    response = await _processor(repository, ollama).process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="The rocket should wobble like in Kerbal Space Program.",
        )
    )

    assert len(response.applied_story_patches) == 1


async def test_an_ambiguous_interest_asks_instead_of_guessing() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient(
        _draft(
            response_text="Kerbal Space Program is a fun one. What should we do with it?",
            interest={"name": "Kerbal Space Program", "ambiguous": True},
        )
    )

    response = await _processor(repository, ollama).process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="Let's add Kerbal Space Program.",
        )
    )

    assert "Is Kerbal Space Program part of this story" in response.response_text
    # The draft's own follow-up gives way, so the child is asked one question.
    assert "What should we do with it?" not in response.response_text
    # Nothing is decided yet, so nothing is written to the story or remembered.
    assert response.applied_story_patches == []
    assert repository.interests == []
    assert all(patch.child_confirmation_required for patch in response.story_patches)


async def test_saying_no_records_the_interest_and_leaves_the_story_alone() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient(
        _draft(interest={"name": "Kerbal Space Program", "ambiguous": True})
    )
    processor = _processor(repository, ollama)

    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Let's add Kerbal Space Program.")
    )
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="no")
    )

    assert [item.name for item in repository.interests] == ["Kerbal Space Program"]
    assert response.applied_story_patches == []
    assert "Kerbal Space Program" in response.response_text


async def test_saying_yes_applies_the_patches_and_still_records_the_interest() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient(
        _draft(interest={"name": "Kerbal Space Program", "ambiguous": True})
    )
    processor = _processor(repository, ollama)

    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Let's add Kerbal Space Program.")
    )
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="yes")
    )

    # Belonging in this story does not make it any less a thing the child loves.
    assert [item.name for item in repository.interests] == ["Kerbal Space Program"]
    assert len(response.applied_story_patches) == 1


async def test_the_clarifying_question_is_spoken_when_streaming() -> None:
    """The voice loop never re-speaks the final text once sentences streamed.

    An appended question would be silently dropped, so it has to reach the
    speaker on its own.
    """
    repository = FakeStoryRepository()
    ollama = StreamingOllamaClient(
        _draft(interest={"name": "Kerbal Space Program", "ambiguous": True})
    )
    spoken: list[str] = []

    async def on_sentence(sentence: str) -> None:
        spoken.append(sentence)

    await _processor(repository, ollama).process(
        AgentTurnRequest(story_id="story_test", transcript="Let's add Kerbal Space Program."),
        on_sentence=on_sentence,
    )

    assert any("Is Kerbal Space Program part of this story" in line for line in spoken)


async def test_known_interests_reach_the_prompt() -> None:
    prompt = build_story_turn_prompt(
        AgentTurnRequest(story_id="story_test", transcript="Make the rocket wobble."),
        TurnIntent.CONTINUE_STORY,
        interests=[
            ChildInterest(
                child_id="local_child",
                name="Kerbal Space Program",
                summary="A game about building wobbly rockets for small green pilots.",
            )
        ],
    )

    assert "Kerbal Space Program" in prompt
    assert "wobbly rockets" in prompt


async def test_an_interest_without_a_description_still_reaches_the_prompt() -> None:
    prompt = build_story_turn_prompt(
        AgentTurnRequest(story_id="story_test", transcript="Make the rocket wobble."),
        TurnIntent.CONTINUE_STORY,
        interests=[ChildInterest(child_id="local_child", name="Kerbal Space Program")],
    )

    assert "Kerbal Space Program" in prompt


class FakeSummaryClient(FakeOllamaClient):
    def __init__(self, summary: str) -> None:
        super().__init__()
        self.summary = summary
        self.prompts: list[str] = []

    async def generate(self, prompt: str, model: str | None = None, **kwargs: object) -> str:
        self.prompts.append(prompt)
        return self.summary


async def test_enrichment_saves_the_description() -> None:
    repository = FakeStoryRepository()
    interest = await repository.save_child_interest(
        ChildInterest(child_id="local_child", name="Kerbal Space Program")
    )
    enricher = InterestEnricher(
        ollama=FakeSummaryClient("A game about building wobbly rockets."),
        repository=repository,
        safety=FakeSafetyReviewer(),
    )

    saved = await enricher.enrich(interest)

    assert saved.summary == "A game about building wobbly rockets."
    assert repository.interests[0].summary == "A game about building wobbly rockets."


async def test_a_model_that_does_not_know_the_interest_writes_no_description() -> None:
    """Local model knowledge is the tradeoff for keeping the child's words home.

    A confident wrong answer about a child's favourite thing is worse than none,
    so the prompt gives the model a way out and it is honoured.
    """
    repository = FakeStoryRepository()
    interest = await repository.save_child_interest(
        ChildInterest(child_id="local_child", name="Glorbo Quest 7")
    )
    enricher = InterestEnricher(
        ollama=FakeSummaryClient("UNKNOWN"),
        repository=repository,
        safety=FakeSafetyReviewer(),
    )

    saved = await enricher.enrich(interest)

    assert saved.summary is None
    assert repository.interests[0].summary is None


async def test_an_unsafe_description_is_discarded() -> None:
    class BlockingReviewer(FakeSafetyReviewer):
        async def review_assistant_output(
            self,
            transcript: str,
            response_text: str,
            story_patches: list[StoryPatch],
        ) -> SafetyDecision:
            return SafetyDecision(
                action=SafetyAction.BLOCK,
                severity=SafetySeverity.REVIEW,
                categories=["assistant_output_review"],
            )

    repository = FakeStoryRepository()
    interest = await repository.save_child_interest(
        ChildInterest(child_id="local_child", name="Something Grim")
    )
    enricher = InterestEnricher(
        ollama=FakeSummaryClient("Something no child should read."),
        repository=repository,
        safety=BlockingReviewer(),
    )

    assert (await enricher.enrich(interest)).summary is None


async def test_the_lookup_runs_in_the_background_and_fills_in_the_description() -> None:
    """The turn does not wait on the lookup, but the interest is described by the next one."""
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient(
        _draft(interest={"name": "Kerbal Space Program", "ambiguous": False})
    )
    summariser = FakeSummaryClient("A game about building wobbly rockets.")
    processor = _processor(
        repository,
        ollama,
        enricher=InterestEnricher(
            ollama=summariser,
            repository=repository,
            safety=FakeSafetyReviewer(),
        ),
    )

    await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="The rocket should wobble like in Kerbal Space Program.",
        )
    )
    # The turn returned before the lookup finished, which is the whole point.
    assert repository.interests[0].summary is None

    await asyncio.gather(*processor._interest_tasks)

    assert repository.interests[0].summary == "A game about building wobbly rockets."
    assert "Kerbal Space Program" in summariser.prompts[0]


async def test_a_described_interest_is_not_looked_up_again() -> None:
    repository = FakeStoryRepository()
    await repository.save_child_interest(
        ChildInterest(
            child_id="local_child",
            name="Kerbal Space Program",
            summary="A game about building wobbly rockets.",
        )
    )
    ollama = FakeOllamaClient(
        _draft(interest={"name": "kerbal space program", "ambiguous": False})
    )
    summariser = FakeSummaryClient("should not be called")
    enricher = InterestEnricher(
        ollama=summariser,
        repository=repository,
        safety=FakeSafetyReviewer(),
    )

    await _processor(repository, ollama, enricher=enricher).process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="More Kerbal Space Program please.",
        )
    )

    # Matched case-insensitively, so it neither duplicates nor re-summarises.
    assert len(repository.interests) == 1
    assert summariser.prompts == []
