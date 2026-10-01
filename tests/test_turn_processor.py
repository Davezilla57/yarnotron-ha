from __future__ import annotations

from datetime import UTC, datetime

from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest, TurnIntent
from yarnotron.models.media import (
    CreateMediaAssetRequest,
    MediaAsset,
    MediaAssetStatus,
    ReviewMediaAssetRequest,
)
from yarnotron.models.interests import ChildInterest, interest_key
from yarnotron.models.memory import ConversationTurn, MemoryChunk, MemorySearchResult
from yarnotron.models.operations import (
    STRIKE_KEY,
    CreateSessionRequest,
    ReleaseSafetyEventRequest,
    ReviewSafetyEventRequest,
    RestoreStoryVersionResponse,
    SafetyEvent,
    StorySession,
    StoryVersion,
)
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import (
    Character,
    CreateStoryRequest,
    Scene,
    Story,
    StoryPatch,
    StoryPatchOperation,
)
from yarnotron.services.memory import StoryMemoryBuilder
from yarnotron.services.story_store import StoryPatchApplier


class FakeOllamaClient:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload
        self.calls = 0

    async def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls += 1
        if self.payload is not None:
            return self.payload
        return {
            "response_text": "The music box reveals a map tucked inside the clockwork.",
            "story_patches": [
                {
                    "operation": "add_fact",
                    "data": {"text": "The clock tower contains a hidden music box."},
                    "child_confirmation_required": False,
                    "rationale": "The child introduced this as durable story content.",
                }
            ],
            "tutor_notes": [],
        }


class FakeStoryRepository:
    def __init__(self) -> None:
        self._story = Story(title="Clock Tower Mystery")
        self.saved_stories: list[Story] = [self._story]
        self._applier = StoryPatchApplier()
        self.created_stories: list[Story] = []
        self.attached_sessions: list[tuple[str, str]] = []
        self.turns: list[ConversationTurn] = []
        self.memory_chunks: list[MemoryChunk] = []
        self.safety_events: list[SafetyEvent] = []
        self.versions: list[StoryVersion] = []
        self.media_assets: list[MediaAsset] = []
        self.interests: list[ChildInterest] = []
        self.sessions: dict[str, StorySession] = {}

    async def create(self, request: CreateStoryRequest) -> Story:
        self._story = Story(title=request.title, synopsis=request.synopsis)
        self.saved_stories.append(self._story)
        self.created_stories.append(self._story)
        return self._story

    async def list(self) -> list[Story]:
        return self.saved_stories

    async def get(self, story_id: str) -> Story | None:
        return self._story

    async def delete(self, story_id: str) -> bool:
        return story_id == self._story.id

    async def apply_patch(self, story_id: str, patch: StoryPatch) -> Story:
        self.versions.insert(
            0,
            StoryVersion(
                id=f"version_{len(self.versions) + 1}",
                story_id=story_id,
                document=self._story.model_copy(deep=True),
                created_at=datetime.now(UTC),
            ),
        )
        self._story = self._applier.apply(self._story, patch)
        return self._story

    async def list_versions(self, story_id: str) -> list[StoryVersion]:
        return self.versions

    async def restore_version(
        self,
        story_id: str,
        version_id: str,
    ) -> RestoreStoryVersionResponse:
        version = next(item for item in self.versions if item.id == version_id)
        self._story = version.document.model_copy(deep=True)
        return RestoreStoryVersionResponse(
            restored_story=self._story,
            previous_version_id="version_previous",
        )

    async def create_session(self, request: CreateSessionRequest) -> StorySession:
        session = StorySession(
            child_id=request.child_id,
            story_id=request.story_id,
            device_id=request.device_id,
        )
        self.sessions[session.id] = session
        return session

    async def attach_session_story(self, session_id: str, story_id: str) -> StorySession:
        self.attached_sessions.append((session_id, story_id))
        session = self.sessions.get(session_id) or StorySession(id=session_id)
        session = session.model_copy(update={"story_id": story_id})
        self.sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> StorySession | None:
        return self.sessions.get(session_id)

    async def latest_session_for_child(self, child_id: str) -> StorySession | None:
        owned = [item for item in self.sessions.values() if item.child_id == child_id]
        return max(owned, key=lambda item: item.started_at) if owned else None

    async def list_sessions(self) -> list[StorySession]:
        return list(self.sessions.values())

    async def end_session(
        self,
        session_id: str,
        ended_at: datetime | None = None,
    ) -> StorySession:
        session = self.sessions.get(session_id) or StorySession(id=session_id)
        session = session.model_copy(update={"ended_at": ended_at or datetime.now(UTC)})
        self.sessions[session_id] = session
        return session

    async def save_conversation_turn(self, turn: ConversationTurn) -> ConversationTurn:
        self.turns.append(turn)
        return turn

    async def list_conversation_turns(self, limit: int = 50) -> list[ConversationTurn]:
        return list(reversed(self.turns))[:limit]

    async def save_safety_event(self, event: SafetyEvent) -> SafetyEvent:
        self.safety_events.append(event)
        if event.session_id is not None:
            # Mirror _refresh_session_safety_summary: the strike count lives on
            # the session, so the processor can read it back next turn.
            session = self.sessions.get(event.session_id) or StorySession(id=event.session_id)
            summary: dict[str, int] = {}
            for saved in self.safety_events:
                if saved.session_id != event.session_id:
                    continue
                action = saved.safety.action.value
                summary[action] = summary.get(action, 0) + 1
                if saved.counted_as_strike:
                    summary[STRIKE_KEY] = summary.get(STRIKE_KEY, 0) + 1
            self.sessions[event.session_id] = session.model_copy(
                update={"safety_summary": summary}
            )
        return event

    async def list_safety_events(self, limit: int = 50) -> list[SafetyEvent]:
        return self.safety_events[-limit:]

    async def review_safety_event(
        self,
        event_id: str,
        request: ReviewSafetyEventRequest,
    ) -> SafetyEvent:
        for index, event in enumerate(self.safety_events):
            if event.id == event_id:
                reviewed = event.model_copy(
                    update={
                        "reviewed_at": datetime.now(UTC),
                        "reviewed_by": request.reviewed_by,
                        "review_note": request.review_note,
                    }
                )
                self.safety_events[index] = reviewed
                return reviewed
        raise KeyError(event_id)

    async def get_safety_event(self, event_id: str) -> SafetyEvent | None:
        return next((e for e in self.safety_events if e.id == event_id), None)

    async def release_safety_event(
        self,
        event_id: str,
        request: ReleaseSafetyEventRequest,
    ) -> SafetyEvent:
        now = datetime.now(UTC)
        return self._update_safety_event(
            event_id,
            {
                "released_at": now,
                "released_by": request.released_by,
                "reviewed_at": now,
                "reviewed_by": request.released_by,
                "review_note": request.review_note,
            },
        )

    async def mark_safety_event_recapped(self, event_id: str) -> SafetyEvent:
        return self._update_safety_event(event_id, {"recap_delivered_at": datetime.now(UTC)})

    def _update_safety_event(self, event_id: str, update: dict) -> SafetyEvent:
        for index, event in enumerate(self.safety_events):
            if event.id == event_id:
                updated = event.model_copy(update=update)
                self.safety_events[index] = updated
                return updated
        raise KeyError(event_id)

    async def save_memory_chunks(self, chunks: list[MemoryChunk]) -> list[MemoryChunk]:
        self.memory_chunks.extend(chunks)
        return chunks

    async def search_memory(
        self,
        query_embedding: list[float],
        story_id: str | None = None,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        return [
            MemorySearchResult(
                id="memory_existing",
                story_id=story_id,
                conversation_turn_id="turn_existing",
                content="Milo found a hidden music box in the clock tower.",
                metadata={"kind": "conversation_turn"},
                score=0.9,
                created_at=datetime.now(UTC),
            )
        ]

    async def delete_memory_chunk(self, memory_id: str) -> bool:
        before = len(self.memory_chunks)
        self.memory_chunks = [chunk for chunk in self.memory_chunks if chunk.id != memory_id]
        return len(self.memory_chunks) < before

    async def create_media_asset(
        self,
        request: CreateMediaAssetRequest,
        safety: SafetyDecision,
        status: MediaAssetStatus = MediaAssetStatus.PENDING_PARENT_REVIEW,
    ) -> MediaAsset:
        asset = MediaAsset(
            child_id=request.child_id,
            story_id=request.story_id,
            session_id=request.session_id,
            conversation_turn_id=request.conversation_turn_id,
            prompt=request.prompt,
            status=status,
            safety=safety,
        )
        self.media_assets.append(asset)
        return asset

    async def list_media_assets(self, limit: int = 50) -> list[MediaAsset]:
        return self.media_assets[-limit:]

    async def get_media_asset(self, media_id: str) -> MediaAsset | None:
        return next((asset for asset in self.media_assets if asset.id == media_id), None)

    async def approve_media_asset(
        self,
        media_id: str,
        request: ReviewMediaAssetRequest,
    ) -> MediaAsset:
        return self._update_media_asset(
            media_id,
            status=MediaAssetStatus.APPROVED,
            parent_approved_at=datetime.now(UTC),
            parent_approved_by=request.reviewed_by,
            parent_note=request.parent_note,
        )

    async def reject_media_asset(
        self,
        media_id: str,
        request: ReviewMediaAssetRequest,
    ) -> MediaAsset:
        return self._update_media_asset(
            media_id,
            status=MediaAssetStatus.REJECTED,
            parent_approved_by=request.reviewed_by,
            parent_note=request.parent_note,
        )

    async def mark_media_asset_generated(
        self,
        media_id: str,
        *,
        asset_path: str,
        asset_url: str,
        mime_type: str,
        generator: str,
        revised_prompt: str | None = None,
    ) -> MediaAsset:
        return self._update_media_asset(
            media_id,
            status=MediaAssetStatus.GENERATED,
            asset_path=asset_path,
            asset_url=asset_url,
            mime_type=mime_type,
            generator=generator,
            revised_prompt=revised_prompt,
        )

    async def mark_media_asset_failed(
        self,
        media_id: str,
        failure_reason: str,
    ) -> MediaAsset:
        return self._update_media_asset(
            media_id,
            status=MediaAssetStatus.FAILED,
            failure_reason=failure_reason,
        )

    async def save_child_interest(self, interest: ChildInterest) -> ChildInterest:
        self.interests = [
            item
            for item in self.interests
            if not (item.child_id == interest.child_id and item.key == interest.key)
        ]
        self.interests.append(interest)
        return interest

    async def get_child_interest(self, child_id: str, name: str) -> ChildInterest | None:
        key = interest_key(name)
        for item in self.interests:
            if item.child_id == child_id and item.key == key:
                return item
        return None

    async def list_child_interests(self, child_id: str, limit: int = 20) -> list[ChildInterest]:
        return [item for item in self.interests if item.child_id == child_id][-limit:]

    def _update_media_asset(self, media_id: str, **updates: object) -> MediaAsset:
        for index, asset in enumerate(self.media_assets):
            if asset.id == media_id:
                updated = asset.model_copy(update=updates)
                self.media_assets[index] = updated
                return updated
        raise KeyError(media_id)


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768


class FakeSafetyReviewer:
    def classify_transcript(self, transcript: str) -> SafetyDecision:
        return SafetyDecision()

    async def review_transcript(
        self,
        transcript: str,
        story: Story | None = None,
        intent: str | None = None,
    ) -> SafetyDecision:
        self.reviewed_with_story = story
        self.reviewed_with_intent = intent
        return SafetyDecision()

    async def review_assistant_output(
        self,
        transcript: str,
        response_text: str,
        story_patches: list[StoryPatch],
    ) -> SafetyDecision:
        return SafetyDecision()


class UnsafeOutputReviewer(FakeSafetyReviewer):
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
            child_message="Let's make that safer before we add it to the story.",
        )


async def test_turn_processor_returns_and_applies_story_patches() -> None:
    repository = FakeStoryRepository()
    processor = TurnProcessor(
        ollama=FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="The inventor hears a tiny music box inside the clock.",
        )
    )

    assert len(response.story_patches) == 1
    assert len(response.applied_story_patches) == 1
    assert response.story_patches[0].operation == StoryPatchOperation.ADD_FACT
    assert response.retrieved_memory_ids == ["memory_existing"]
    assert response.updated_story is not None
    assert response.updated_story.facts[0].text == "The clock tower contains a hidden music box."
    assert len(repository.turns) == 1
    assert len(repository.memory_chunks) == 1
    assert response.memory_chunk_ids == [repository.memory_chunks[0].id]
    assert repository.memory_chunks[0].embedding == [0.1] * 768


async def test_turn_processor_creates_story_for_first_story_turn() -> None:
    repository = FakeStoryRepository()
    processor = TurnProcessor(
        ollama=FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            session_id="session_test",
            transcript="Let's start a story called The Moon Clock. Milo finds a glowing key.",
        )
    )

    assert response.updated_story is not None
    assert response.updated_story.title == "The Moon Clock"
    assert response.response_text.startswith("New story: The Moon Clock.")
    assert "Remember that title" in response.response_text
    assert repository.created_stories[0].title == "The Moon Clock"
    assert repository.attached_sessions == [("session_test", response.updated_story.id)]
    assert repository.turns[0].story_id == response.updated_story.id
    assert len(repository.memory_chunks) == 1


async def test_turn_processor_prompts_when_first_turn_might_be_new_or_previous() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            session_id="session_test",
            transcript="Let's keep going with the story.",
        )
    )

    assert ollama.calls == 0
    assert response.updated_story is None
    assert response.story_patches == []
    assert "new story" in response.response_text
    assert "continuing one we already started" in response.response_text
    assert repository.created_stories == []
    assert repository.attached_sessions == []
    assert repository.turns[0].story_id is None


async def test_turn_processor_lists_available_story_titles_only() -> None:
    repository = FakeStoryRepository()
    moon_story = Story(id="story_the_moon_clock", title="The Moon Clock")
    repository.saved_stories.append(moon_story)
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            session_id="session_test",
            transcript="What stories are available?",
        )
    )

    assert ollama.calls == 0
    assert response.response_text == (
        "Available stories: Clock Tower Mystery; The Moon Clock. "
        "Which one would you like to pick?"
    )
    assert "story_the_moon_clock" not in response.response_text
    assert response.updated_story is None
    assert repository.turns[0].story_id is None


async def test_turn_processor_picks_story_from_title_response_without_ollama() -> None:
    repository = FakeStoryRepository()
    moon_story = Story(id="story_the_moon_clock", title="The Moon Clock")
    repository.saved_stories.append(moon_story)
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            session_id="session_test",
            transcript="The Moon Clock",
        )
    )

    assert ollama.calls == 0
    assert response.updated_story == moon_story
    assert response.response_text == "I picked The Moon Clock. What part should we continue?"
    assert repository.created_stories == []
    assert repository.attached_sessions == [("session_test", moon_story.id)]
    assert repository.turns[0].story_id == moon_story.id
    assert len(repository.memory_chunks) == 1


async def test_turn_processor_can_switch_active_story_from_title_response() -> None:
    repository = FakeStoryRepository()
    moon_story = Story(id="story_the_moon_clock", title="The Moon Clock")
    repository.saved_stories.append(moon_story)
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            story_id=repository._story.id,
            session_id="session_test",
            transcript="The Moon Clock",
        )
    )

    assert ollama.calls == 0
    assert response.updated_story == moon_story
    assert repository.attached_sessions == [("session_test", moon_story.id)]
    assert repository.turns[0].story_id == moon_story.id


async def test_turn_processor_resumes_existing_story_by_title_on_first_turn() -> None:
    repository = FakeStoryRepository()
    processor = TurnProcessor(
        ollama=FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            session_id="session_test",
            transcript="Continue Clock Tower Mystery with a secret door.",
        )
    )

    assert response.updated_story is not None
    assert response.updated_story.id == repository._story.id
    assert response.response_text.startswith("Back to Clock Tower Mystery.")
    assert repository.created_stories == []
    assert repository.attached_sessions == [("session_test", repository._story.id)]
    assert repository.turns[0].story_id == repository._story.id


async def test_turn_processor_does_not_apply_patches_when_output_review_redirects() -> None:
    repository = FakeStoryRepository()
    processor = TurnProcessor(
        ollama=FakeOllamaClient(),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=UnsafeOutputReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="The inventor hears a tiny music box inside the clock.",
        )
    )

    assert response.safety.action == SafetyAction.REDIRECT
    assert response.response_text == "Let's make that safer before we add it to the story."
    assert response.applied_story_patches == []
    assert repository._story.facts == []
    assert len(repository.safety_events) == 1


async def test_turn_processor_requires_confirmation_for_destructive_edits() -> None:
    repository = FakeStoryRepository()
    repository._story.scenes.append(
        Scene(id="scene_tower", title="Clock Tower", summary="Milo climbs the tower.")
    )
    processor = TurnProcessor(
        ollama=FakeOllamaClient(
            {
                "response_text": "I can remove that scene.",
                "story_patches": [
                    {
                        "operation": "remove_scene",
                        "data": {"id": "scene_tower"},
                        "child_confirmation_required": False,
                        "rationale": "The child asked to remove the scene.",
                    }
                ],
                "tutor_notes": [],
            }
        ),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="Remove the clock tower scene.",
        )
    )

    assert response.story_patches[0].child_confirmation_required is True
    assert response.applied_story_patches == []
    assert [scene.id for scene in repository._story.scenes] == ["scene_tower"]
    assert "Should I make it?" in response.response_text
    # The prompt is appended to a spoken reply, so it has to be brief. The old
    # wording was 107 characters -- about five seconds, as long as an entire
    # reply is supposed to be.
    appended = response.response_text.split("I can remove that scene.")[-1].strip()
    assert len(appended) < 45, appended


async def test_turn_processor_merges_duplicate_characters() -> None:
    repository = FakeStoryRepository()
    repository._story.characters.append(
        Character(id="character_milo", name="Milo", role="inventor", traits=["curious"])
    )
    processor = TurnProcessor(
        ollama=FakeOllamaClient(
            {
                "response_text": "Milo is getting braver.",
                "story_patches": [
                    {
                        "operation": "add_character",
                        "data": {"name": "milo", "traits": ["brave"], "notes": "He trusts maps."},
                        "child_confirmation_required": False,
                        "rationale": "The child added more about Milo.",
                    }
                ],
                "tutor_notes": [],
            }
        ),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="Milo is brave now and he trusts maps.",
        )
    )

    assert response.applied_story_patches[0].operation == StoryPatchOperation.UPDATE_CHARACTER
    assert len(repository._story.characters) == 1
    assert repository._story.characters[0].traits == ["curious", "brave"]
    assert repository._story.characters[0].notes == "He trusts maps."


async def test_turn_processor_preserves_child_wording_for_new_scenes() -> None:
    repository = FakeStoryRepository()
    transcript = "Milo whispered that the moonlight clock was singing under the floor."
    processor = TurnProcessor(
        ollama=FakeOllamaClient(
            {
                "response_text": "That sounds mysterious.",
                "story_patches": [
                    {
                        "operation": "add_scene",
                        "data": {
                            "title": "The Singing Clock",
                            "summary": "Milo hears the clock beneath the floor.",
                            "sequence": 1,
                        },
                        "child_confirmation_required": False,
                        "rationale": "The child added a new scene.",
                    }
                ],
                "tutor_notes": [],
            }
        ),
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    await processor.process(AgentTurnRequest(story_id="story_test", transcript=transcript))

    assert repository._story.scenes[0].draft_text == transcript


async def test_turn_processor_undo_restores_latest_version_without_ollama() -> None:
    repository = FakeStoryRepository()
    await repository.apply_patch(
        "story_test",
        StoryPatch(
            operation=StoryPatchOperation.UPDATE_TITLE,
            data={"title": "Clock Tower Trouble"},
        ),
    )
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Undo last change.")
    )

    assert ollama.calls == 0
    assert response.intent.value == "undo_last_change"
    assert response.updated_story is not None
    assert response.updated_story.title == "Clock Tower Mystery"


async def test_turn_processor_creates_media_request_without_ollama() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(
            story_id="story_test",
            transcript="Draw a picture of Milo beside the moonlight clock.",
        )
    )

    assert ollama.calls == 0
    assert response.intent.value == "create_image"
    assert len(repository.media_assets) == 1
    assert repository.media_assets[0].status == MediaAssetStatus.PENDING_PARENT_REVIEW
    assert repository.media_assets[0].conversation_turn_id == response.turn_id
    assert response.media_asset_ids == [repository.media_assets[0].id]
    assert "that would make a cool picture" in response.response_text
    assert "Milo beside the moonlight clock" in response.response_text


async def test_turn_processor_returns_unknown_without_ollama_for_unclear_story_request() -> None:
    repository = FakeStoryRepository()
    ollama = FakeOllamaClient()
    processor = TurnProcessor(
        ollama=ollama,
        repository=repository,
        embedder=FakeEmbedder(),
        memory_builder=StoryMemoryBuilder(),
        safety=FakeSafetyReviewer(),
    )

    response = await processor.process(
        AgentTurnRequest(story_id="story_test", transcript="Huh?")
    )

    assert ollama.calls == 0
    assert response.intent == TurnIntent.UNKNOWN
    assert response.story_patches == []
    assert response.applied_story_patches == []
    assert "did not quite catch" in response.response_text
    assert repository.turns[0].intent == TurnIntent.UNKNOWN
