from __future__ import annotations

import pytest

from yarnotron.agent.turn_processor import TurnProcessor, _confirmation_answer
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.services.memory import StoryMemoryBuilder

from tests.test_turn_processor import (
    FakeEmbedder,
    FakeOllamaClient,
    FakeSafetyReviewer,
    FakeStoryRepository,
)

REMOVE_DRAFT = {
    "response_text": "Are you sure you want Fennick gone?",
    "story_patches": [
        {
            "operation": "remove_character",
            "data": {"id": "character_fennick"},
            "child_confirmation_required": True,
            "rationale": "Destructive edit.",
        }
    ],
    "tutor_notes": [],
}


def build(payload):
    repository = FakeStoryRepository()
    return (
        TurnProcessor(
            ollama=FakeOllamaClient(payload),
            repository=repository,
            embedder=FakeEmbedder(),
            memory_builder=StoryMemoryBuilder(),
            safety=FakeSafetyReviewer(),
        ),
        repository,
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", True), ("Yes!", True), ("yeah", True), ("do it", True),
        ("go ahead", True), ("Yes, do it.", True),
        ("no", False), ("Nope.", False), ("never mind", False), ("leave it", False),
        # Anything that is not plainly an answer must not be treated as one:
        # a false positive here applies a destructive edit unasked.
        ("yes and then she ran away", None),
        ("no one was there", None),
        ("okay so the seal pup swims off", None),
        ("Wren says yes to the adventure", None),
        ("", None),
    ],
)
def test_confirmation_answer_is_narrow(text, expected):
    assert _confirmation_answer(text) is expected


async def test_destructive_patch_is_not_applied_until_confirmed():
    processor, repository = build(REMOVE_DRAFT)
    first = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Remove Fennick.")
    )
    assert first.story_patches and not first.applied_story_patches


async def test_saying_yes_applies_the_pending_change():
    """This is the flow that could not complete before.

    The policy flagged the patch, the prompt asked, and nothing recorded what
    was pending -- so "yes" routed to UNKNOWN and the change was dropped for
    good.
    """
    processor, repository = build(REMOVE_DRAFT)
    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Remove Fennick.")
    )
    second = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="yes")
    )

    assert [p.operation.value for p in second.applied_story_patches] == ["remove_character"]
    assert not second.applied_story_patches[0].child_confirmation_required
    assert "made that change" in second.response_text


async def test_saying_no_discards_the_pending_change():
    processor, repository = build(REMOVE_DRAFT)
    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Remove Fennick.")
    )
    second = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="no")
    )

    assert second.applied_story_patches == []
    assert "left the story exactly as it was" in second.response_text


async def test_confirming_twice_does_not_reapply():
    processor, repository = build(REMOVE_DRAFT)
    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Remove Fennick.")
    )
    await processor.process(AgentTurnRequest(story_id="story_test", transcript="yes"))
    third = await processor.process(AgentTurnRequest(story_id="story_test", transcript="yes"))

    # The confirming turn recorded the patch as applied, so nothing is pending.
    assert third.applied_story_patches == []


async def test_unrelated_reply_after_a_prompt_is_a_normal_turn():
    """The child ignoring the question must not be read as an answer."""
    processor, repository = build(REMOVE_DRAFT)
    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Remove Fennick.")
    )
    second = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="The seal pup swims away instead.")
    )

    assert second.intent.value == "continue_story"
    assert second.applied_story_patches == [] or all(
        p.operation.value != "remove_character" for p in second.applied_story_patches
    )


async def test_yes_with_nothing_pending_is_a_normal_turn():
    processor, repository = build(
        {"response_text": "Tell me more!", "story_patches": [], "tutor_notes": []}
    )
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="yes")
    )
    # Routes as it always did rather than confirming something that never existed.
    assert response.intent.value in {"unknown", "continue_story"}
