from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from yarnotron.models.interests import InterestCandidate
from yarnotron.models.safety import SafetyDecision
from yarnotron.models.story import Story, StoryPatch


class TurnIntent(StrEnum):
    CONTINUE_STORY = "continue_story"
    EDIT_STORY = "edit_story"
    ASK_DEFINITION = "ask_definition"
    ASK_GRAMMAR = "ask_grammar"
    CREATE_IMAGE = "create_image"
    PARENT_COMMAND = "parent_command"
    UNDO_LAST_CHANGE = "undo_last_change"
    UNKNOWN = "unknown"


class AgentTurnRequest(BaseModel):
    child_id: str = "local_child"
    story_id: str | None = None
    session_id: str | None = None
    device_id: str | None = None
    transcript: str = Field(min_length=1, max_length=10_000)
    model: str | None = None


class TutorNote(BaseModel):
    kind: str
    text: str


class AgentTurnResponse(BaseModel):
    turn_id: str = Field(default_factory=lambda: f"turn_{uuid4().hex}")
    # The session this turn landed in. Echoed back so a caller that sent none
    # can keep the next turn in the same one -- safety strikes are counted per
    # session, so continuity is what makes them add up.
    session_id: str | None = None
    intent: TurnIntent
    response_text: str
    safety: SafetyDecision
    story_patches: list[StoryPatch] = Field(default_factory=list)
    applied_story_patches: list[StoryPatch] = Field(default_factory=list)
    updated_story: Story | None = None
    tutor_notes: list[TutorNote] = Field(default_factory=list)
    memory_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    media_asset_ids: list[str] = Field(default_factory=list)


class AgentDraft(BaseModel):
    # Empty is tolerated so a draft with no usable text still produces a turn;
    # TurnProcessor substitutes _fallback_response for the spoken reply.
    response_text: str = ""
    story_patches: list[StoryPatch] = Field(default_factory=list)
    tutor_notes: list[TutorNote] = Field(default_factory=list)
    interest_candidate: InterestCandidate | None = None
