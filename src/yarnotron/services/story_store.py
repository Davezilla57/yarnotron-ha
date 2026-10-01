"""The storage contract, and the pieces that go with it.

Kept separate from the SQLite implementation so a route or an agent can say
what it needs from a repository without pulling the whole datastore in.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol

from fastapi import Request

from yarnotron.models.media import (
    CreateMediaAssetRequest,
    MediaAsset,
    MediaAssetStatus,
    ReviewMediaAssetRequest,
)
from yarnotron.models.interests import ChildInterest
from yarnotron.models.memory import ConversationTurn, MemoryChunk, MemorySearchResult
from yarnotron.models.operations import (
    CreateSessionRequest,
    ReleaseSafetyEventRequest,
    RestoreStoryVersionResponse,
    ReviewSafetyEventRequest,
    SafetyEvent,
    StorySession,
    StoryVersion,
)
from yarnotron.models.safety import SafetyDecision
from yarnotron.models.story import (
    Character,
    CreateStoryRequest,
    Scene,
    Story,
    StoryFact,
    StoryPatch,
    StoryPatchOperation,
)


class StoryRepository(Protocol):
    async def create(self, request: CreateStoryRequest) -> Story: ...

    async def list(self) -> list[Story]: ...

    async def get(self, story_id: str) -> Story | None: ...

    async def delete(self, story_id: str) -> bool: ...

    async def apply_patch(self, story_id: str, patch: StoryPatch) -> Story: ...

    async def list_versions(self, story_id: str) -> list[StoryVersion]: ...

    async def restore_version(
        self,
        story_id: str,
        version_id: str,
    ) -> RestoreStoryVersionResponse: ...

    async def create_session(self, request: CreateSessionRequest) -> StorySession: ...

    async def attach_session_story(self, session_id: str, story_id: str) -> StorySession: ...

    async def get_session(self, session_id: str) -> StorySession | None: ...

    async def latest_session_for_child(self, child_id: str) -> StorySession | None: ...

    async def list_sessions(self) -> list[StorySession]: ...

    async def end_session(
        self,
        session_id: str,
        ended_at: datetime | None = None,
    ) -> StorySession: ...

    async def save_conversation_turn(self, turn: ConversationTurn) -> ConversationTurn: ...

    async def list_conversation_turns(self, limit: int = 50) -> list[ConversationTurn]: ...

    async def save_memory_chunks(self, chunks: list[MemoryChunk]) -> list[MemoryChunk]: ...

    async def save_safety_event(self, event: SafetyEvent) -> SafetyEvent: ...

    async def list_safety_events(self, limit: int = 50) -> list[SafetyEvent]: ...

    async def review_safety_event(
        self,
        event_id: str,
        request: ReviewSafetyEventRequest,
    ) -> SafetyEvent: ...

    async def get_safety_event(self, event_id: str) -> SafetyEvent | None: ...

    async def release_safety_event(
        self,
        event_id: str,
        request: ReleaseSafetyEventRequest,
    ) -> SafetyEvent: ...

    async def mark_safety_event_recapped(self, event_id: str) -> SafetyEvent: ...

    async def search_memory(
        self,
        query_embedding: list[float],
        story_id: str | None = None,
        limit: int = 5,
    ) -> list[MemorySearchResult]: ...

    async def delete_memory_chunk(self, memory_id: str) -> bool: ...

    async def create_media_asset(
        self,
        request: CreateMediaAssetRequest,
        safety: SafetyDecision,
        status: MediaAssetStatus = MediaAssetStatus.PENDING_PARENT_REVIEW,
    ) -> MediaAsset: ...

    async def list_media_assets(self, limit: int = 50) -> list[MediaAsset]: ...

    async def get_media_asset(self, media_id: str) -> MediaAsset | None: ...

    async def approve_media_asset(
        self,
        media_id: str,
        request: ReviewMediaAssetRequest,
    ) -> MediaAsset: ...

    async def reject_media_asset(
        self,
        media_id: str,
        request: ReviewMediaAssetRequest,
    ) -> MediaAsset: ...

    async def mark_media_asset_generated(
        self,
        media_id: str,
        *,
        asset_path: str,
        asset_url: str,
        mime_type: str,
        generator: str,
        revised_prompt: str | None = None,
    ) -> MediaAsset: ...

    async def mark_media_asset_failed(
        self,
        media_id: str,
        failure_reason: str,
    ) -> MediaAsset: ...

    async def save_child_interest(self, interest: ChildInterest) -> ChildInterest: ...

    async def get_child_interest(self, child_id: str, name: str) -> ChildInterest | None: ...

    async def list_child_interests(
        self,
        child_id: str,
        limit: int = 20,
    ) -> list[ChildInterest]: ...


def story_id_from_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.casefold()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "untitled_story"
    return f"story_{slug[:80].strip('_') or 'untitled_story'}"


class StoryPatchApplier:
    def apply(self, story: Story, patch: StoryPatch) -> Story:
        updated = story.model_copy(deep=True)
        data = patch.data

        match patch.operation:
            case StoryPatchOperation.UPDATE_TITLE:
                updated.title = _required_str(data, "title")
            case StoryPatchOperation.UPDATE_SYNOPSIS:
                updated.synopsis = _optional_str(data, "synopsis")
            case StoryPatchOperation.ADD_CHARACTER:
                updated.characters.append(Character.model_validate(data))
            case StoryPatchOperation.UPDATE_CHARACTER:
                character_id = _required_str(data, "id")
                character = _find_by_id(updated.characters, character_id)
                character.name = str(data.get("name", character.name))
                character.role = _optional_str(data, "role", character.role)
                if "traits" in data and isinstance(data["traits"], list):
                    character.traits = [str(item) for item in data["traits"]]
                character.notes = _optional_str(data, "notes", character.notes)
            case StoryPatchOperation.REMOVE_CHARACTER:
                character_id = _required_str(data, "id")
                updated.characters = [item for item in updated.characters if item.id != character_id]
            case StoryPatchOperation.ADD_SCENE:
                updated.scenes.append(Scene.model_validate(data))
                updated.scenes.sort(key=lambda item: item.sequence)
            case StoryPatchOperation.UPDATE_SCENE:
                scene_id = _required_str(data, "id")
                scene = _find_by_id(updated.scenes, scene_id)
                scene.title = str(data.get("title", scene.title))
                scene.summary = str(data.get("summary", scene.summary))
                scene.draft_text = _optional_str(data, "draft_text", scene.draft_text)
                if "sequence" in data:
                    scene.sequence = int(data["sequence"])
                updated.scenes.sort(key=lambda item: item.sequence)
            case StoryPatchOperation.REMOVE_SCENE:
                scene_id = _required_str(data, "id")
                updated.scenes = [item for item in updated.scenes if item.id != scene_id]
            case StoryPatchOperation.ADD_FACT:
                updated.facts.append(StoryFact.model_validate(data))
            case StoryPatchOperation.REMOVE_FACT:
                fact_id = _required_str(data, "id")
                updated.facts = [item for item in updated.facts if item.id != fact_id]

        updated.updated_at = datetime.now(UTC)
        return updated


def _required_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Patch data must include a non-empty string field: {key}")
    return value


def _optional_str(data: dict[str, object], key: str, default: str | None = None) -> str | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Patch data field must be a string when present: {key}")
    return value


def _find_by_id(items: list[Character] | list[Scene], item_id: str) -> Character | Scene:
    for item in items:
        if item.id == item_id:
            return item
    raise ValueError(f"Unable to find item with id: {item_id}")


def get_story_repository(request: Request) -> StoryRepository:
    """The repository opened by the app's lifespan."""
    repository = getattr(request.app.state, "story_repository", None)
    if repository is None:  # pragma: no cover - only if lifespan did not run
        raise RuntimeError("story repository is not initialised")
    return repository
