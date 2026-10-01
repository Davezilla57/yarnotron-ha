from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from yarnotron.models.safety import SafetyDecision


class MediaAssetStatus(StrEnum):
    PENDING_PARENT_REVIEW = "pending_parent_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    GENERATED = "generated"
    FAILED = "failed"


class MediaAsset(BaseModel):
    id: str = Field(default_factory=lambda: f"media_{uuid4().hex}")
    child_id: str = "local_child"
    story_id: str | None = None
    session_id: str | None = None
    conversation_turn_id: str | None = None
    prompt: str
    revised_prompt: str | None = None
    status: MediaAssetStatus = MediaAssetStatus.PENDING_PARENT_REVIEW
    safety: SafetyDecision
    parent_approved_at: datetime | None = None
    parent_approved_by: str | None = None
    parent_note: str | None = None
    asset_path: str | None = None
    asset_url: str | None = None
    mime_type: str | None = None
    generator: str | None = None
    failure_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CreateMediaAssetRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    child_id: str = "local_child"
    story_id: str | None = None
    session_id: str | None = None
    conversation_turn_id: str | None = None


class ReviewMediaAssetRequest(BaseModel):
    reviewed_by: str = "local_parent"
    parent_note: str | None = None


class GenerateMediaAssetRequest(BaseModel):
    generator: str = "local_svg_placeholder"
