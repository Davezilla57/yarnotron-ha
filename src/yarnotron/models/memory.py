from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from yarnotron.models.agent import TurnIntent, TutorNote
from yarnotron.models.safety import SafetyDecision
from yarnotron.models.story import StoryPatch


class ConversationTurn(BaseModel):
    id: str = Field(default_factory=lambda: f"turn_{uuid4().hex}")
    child_id: str
    story_id: str | None = None
    session_id: str | None = None
    device_id: str | None = None
    intent: TurnIntent
    transcript: str
    response_text: str
    safety: SafetyDecision
    story_patches: list[StoryPatch] = Field(default_factory=list)
    applied_story_patches: list[StoryPatch] = Field(default_factory=list)
    tutor_notes: list[TutorNote] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"memory_{uuid4().hex}")
    story_id: str | None = None
    conversation_turn_id: str | None = None
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    story_id: str | None = None
    limit: int = Field(default=5, ge=1, le=25)


class MemorySearchResult(BaseModel):
    id: str
    story_id: str | None = None
    conversation_turn_id: str | None = None
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float
    created_at: datetime
