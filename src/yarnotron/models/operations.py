from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from yarnotron.models.safety import SafetyDecision
from yarnotron.models.story import Story


# Key under which StorySession.safety_summary carries the strike count. The
# other keys are SafetyAction values, so this must not collide with one.
STRIKE_KEY = "strike"


class StorySession(BaseModel):
    id: str = Field(default_factory=lambda: f"session_{uuid4().hex}")
    child_id: str = "local_child"
    story_id: str | None = None
    device_id: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    safety_summary: dict[str, int] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    child_id: str = "local_child"
    story_id: str | None = None
    device_id: str | None = None


class EndSessionRequest(BaseModel):
    ended_at: datetime | None = None


class ReviewSafetyEventRequest(BaseModel):
    reviewed_by: str = "local_parent"
    review_note: str | None = None


class ReleaseSafetyEventRequest(BaseModel):
    released_by: str = "local_parent"
    review_note: str | None = None


class StoryVersion(BaseModel):
    id: str
    story_id: str
    document: Story
    created_at: datetime


class RestoreStoryVersionResponse(BaseModel):
    restored_story: Story
    previous_version_id: str


class SafetyEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"safety_{uuid4().hex}")
    child_id: str
    story_id: str | None = None
    session_id: str | None = None
    conversation_turn_id: str | None = None
    safety: SafetyDecision
    transcript_excerpt: str
    parent_alert_sent: bool = False
    # True when this event counted toward the session's strike limit. Only what
    # the child said counts; a redirect caused by the assistant's own output is
    # recorded but is not the child's strike.
    counted_as_strike: bool = False
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    review_note: str | None = None
    # A parent decided this block was wrong and let the turn through. The
    # transcript stays allowed for the rest of that session, so the child can
    # revisit the idea without it being stopped again.
    released_at: datetime | None = None
    released_by: str | None = None
    # The child was not listening when the released turn ran, so the next time
    # they open the story they are told what landed. Stamped once delivered.
    recap_delivered_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
