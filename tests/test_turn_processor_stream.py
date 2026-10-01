from __future__ import annotations

import json

from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import StoryPatch
from yarnotron.services.memory import StoryMemoryBuilder
from yarnotron.services.ollama import StreamEvent

from tests.test_turn_processor import (
    FakeEmbedder,
    FakeOllamaClient,
    FakeSafetyReviewer,
    FakeStoryRepository,
)

DRAFT = {
    "response_text": "Oh, a seal pup! I added that to the story. What does Wren do next?",
    "story_patches": [
        {
            "operation": "add_fact",
            "data": {"text": "A seal pup was caught in the netting."},
            "child_confirmation_required": False,
            "rationale": "The child added durable story content.",
        }
    ],
    "tutor_notes": [],
}


class StreamingOllamaClient(FakeOllamaClient):
    """Serves a draft the way Ollama does: fenced, in small chunks."""

    def __init__(self, payload: dict[str, object] | None = None, chunk_size: int = 5) -> None:
        super().__init__(payload)
        self.chunk_size = chunk_size
        self.streamed = 0

    async def stream_json(self, prompt: str, model: str | None = None, **kwargs: object):
        self.streamed += 1
        body = "```json\n" + json.dumps(self.payload or DRAFT) + "\n```"
        from yarnotron.services.incremental_json import JsonStringFieldReader

        reader = JsonStringFieldReader("response_text")
        for i in range(0, len(body), self.chunk_size):
            was_done = reader.done
            text = reader.feed(body[i : i + self.chunk_size])
            if not was_done and reader.done:
                yield StreamEvent(text=text, field_done=True)
            elif text:
                yield StreamEvent(text=text)
        yield StreamEvent(payload=self.payload or DRAFT, stats={})


def build(ollama=None, safety=None, repository=None):
    return (
        TurnProcessor(
            ollama=ollama or StreamingOllamaClient(),
            repository=repository or FakeStoryRepository(),
            embedder=FakeEmbedder(),
            memory_builder=StoryMemoryBuilder(),
            safety=safety or FakeSafetyReviewer(),
        ),
        repository,
    )


def collector(into: list[str]):
    async def sink(sentence: str) -> None:
        into.append(sentence)
    return sink


async def run(processor, sink, transcript="It is a seal pup stuck in the netting."):
    return await processor.process(
        AgentTurnRequest(story_id="story_test", transcript=transcript),
        on_sentence=sink,
    )


async def test_sentences_are_delivered_while_streaming():
    spoken: list[str] = []
    processor, _ = build()
    response = await run(processor, collector(spoken))

    assert spoken == [
        "Oh, a seal pup!",
        "I added that to the story.",
        "What does Wren do next?",
    ]
    assert response.response_text == DRAFT["response_text"]


async def test_streaming_and_buffered_produce_the_same_response():
    """The exit criterion: streaming must not change what the turn produces."""
    buffered_repo = FakeStoryRepository()
    streamed_repo = FakeStoryRepository()

    buffered = TurnProcessor(
        ollama=FakeOllamaClient(DRAFT),
        repository=buffered_repo,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )
    streamed = TurnProcessor(
        ollama=StreamingOllamaClient(DRAFT),
        repository=streamed_repo,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    request = AgentTurnRequest(story_id="story_test", transcript="It is a seal pup.")
    a = await buffered.process(request)
    b = await streamed.process(request, on_sentence=lambda s: _noop())

    # Each fake repository mints its own random ids, so compare substance.
    # `session_id` is one of them: neither call named a session, so each side
    # auto-created its own.
    ignore = {"turn_id", "memory_chunk_ids", "session_id"}

    def comparable(response):
        data = {k: v for k, v in response.model_dump(mode="json").items() if k not in ignore}
        story = data.get("updated_story")
        if story:
            story.pop("id", None)
            story.pop("created_at", None)
            story.pop("updated_at", None)
            for fact in story.get("facts", []):
                fact.pop("id", None)
        return data

    assert comparable(a) == comparable(b)
    assert len(buffered_repo.turns) == len(streamed_repo.turns) == 1


async def _noop():
    return None


class VetoSecondSentence(FakeSafetyReviewer):
    """Allows the first sentence, blocks everything after it."""

    def __init__(self) -> None:
        self.reviews = 0

    async def review_assistant_output(
        self,
        transcript: str,
        response_text: str,
        story_patches: list[StoryPatch],
    ) -> SafetyDecision:
        self.reviews += 1
        if self.reviews <= 1:
            return SafetyDecision()
        return SafetyDecision(
            action=SafetyAction.REDIRECT,
            severity=SafetySeverity.REVIEW,
            categories=["assistant_output_review"],
            child_message="Let's steer that somewhere safer.",
        )


async def test_a_vetoed_sentence_is_never_spoken():
    spoken: list[str] = []
    processor, _ = build(safety=VetoSecondSentence())
    await run(processor, collector(spoken))

    assert spoken == ["Oh, a seal pup!"]


async def test_gate_stops_all_later_sentences_once_it_vetoes():
    spoken: list[str] = []
    processor, _ = build(safety=VetoSecondSentence())
    await run(processor, collector(spoken))

    # Not merely the offending sentence -- nothing after the veto is spoken.
    assert len(spoken) == 1


async def test_patches_are_applied_after_the_reply_is_spoken():
    """Deferring persistence past first audio is the point of streaming."""
    order: list[str] = []
    repository = FakeStoryRepository()

    async def sink(sentence: str) -> None:
        order.append("spoke")

    original = repository.apply_patch

    async def traced(story_id, patch):
        order.append("applied")
        return await original(story_id, patch)

    repository.apply_patch = traced
    processor = TurnProcessor(
        ollama=StreamingOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )
    await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="It is a seal pup."),
        on_sentence=sink,
    )

    assert "spoke" in order and "applied" in order
    assert order.index("spoke") < order.index("applied")


async def test_buffered_path_never_calls_stream():
    ollama = StreamingOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=FakeStoryRepository(),
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )
    await processor.process(AgentTurnRequest(story_id="story_test", transcript="A seal pup."))
    assert ollama.streamed == 0
    assert ollama.calls == 1


BAD_OPERATION_DRAFT = {
    "response_text": "Mackerel is a wonderful name! What does she call out first?",
    "story_patches": [
        {"operation": "rename_character", "data": {"id": "c1", "name": "Mackerel"}},
        {
            "operation": "add_fact",
            "data": {"text": "The seal pup is named Mackerel."},
            "child_confirmation_required": False,
            "rationale": "The child named the pup.",
        },
    ],
    "tutor_notes": [],
}


async def test_one_hallucinated_patch_operation_does_not_lose_the_turn():
    """Reproduces the seeded t16 failure.

    gemma4 emitted a patch with an operation outside the enum. Validating the
    whole draft at once turned that into a 500 for the entire turn -- and in the
    streaming path the reply had already been spoken, so the child heard it and
    then got an error.
    """
    spoken: list[str] = []
    processor, _ = build(ollama=StreamingOllamaClient(BAD_OPERATION_DRAFT))
    response = await run(processor, collector(spoken))

    assert response.response_text.startswith("Mackerel is a wonderful name!")
    assert spoken  # the reply was delivered rather than lost
    # The bad patch is dropped; the good one survives.
    assert [p.operation.value for p in response.story_patches] == ["add_fact"]


async def test_bad_patch_is_dropped_on_the_buffered_path_too():
    processor = TurnProcessor(
        ollama=FakeOllamaClient(BAD_OPERATION_DRAFT),
        repository=FakeStoryRepository(),
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Name her Mackerel.")
    )
    assert [p.operation.value for p in response.story_patches] == ["add_fact"]


async def test_draft_with_no_text_falls_back_instead_of_failing():
    draft = {"response_text": "", "story_patches": [], "tutor_notes": []}
    processor = TurnProcessor(
        ollama=FakeOllamaClient(draft),
        repository=FakeStoryRepository(),
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )
    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Keep going.")
    )
    assert response.response_text  # a fallback line, not an empty reply
