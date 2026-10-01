from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class Character(BaseModel):
    id: str = Field(default_factory=lambda: f"character_{uuid4().hex}")
    name: str
    role: str | None = None
    traits: list[str] = Field(default_factory=list)
    notes: str | None = None


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: f"scene_{uuid4().hex}")
    title: str
    summary: str
    draft_text: str | None = None
    sequence: int = 0


class StoryFact(BaseModel):
    id: str = Field(default_factory=lambda: f"fact_{uuid4().hex}")
    text: str
    source_turn_id: str | None = None


class Story(BaseModel):
    id: str = Field(default_factory=lambda: f"story_{uuid4().hex}")
    title: str
    synopsis: str | None = None
    characters: list[Character] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    facts: list[StoryFact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CreateStoryRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    synopsis: str | None = Field(default=None, max_length=2_000)


class StoryPatchOperation(StrEnum):
    UPDATE_TITLE = "update_title"
    UPDATE_SYNOPSIS = "update_synopsis"
    ADD_CHARACTER = "add_character"
    UPDATE_CHARACTER = "update_character"
    REMOVE_CHARACTER = "remove_character"
    ADD_SCENE = "add_scene"
    UPDATE_SCENE = "update_scene"
    REMOVE_SCENE = "remove_scene"
    ADD_FACT = "add_fact"
    REMOVE_FACT = "remove_fact"


class StoryPatch(BaseModel):
    operation: StoryPatchOperation
    data: dict[str, object] = Field(default_factory=dict)
    child_confirmation_required: bool = False
    rationale: str | None = None

