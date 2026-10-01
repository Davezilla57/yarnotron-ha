from enum import StrEnum

from pydantic import BaseModel, Field


class SafetyAction(StrEnum):
    ALLOW = "allow"
    REDIRECT = "redirect"
    BLOCK = "block"
    ALERT_PARENT = "alert_parent"


class SafetySeverity(StrEnum):
    NONE = "none"
    INFO = "info"
    REVIEW = "review"
    ALERT = "alert"
    URGENT = "urgent"


class SafetyDecision(BaseModel):
    action: SafetyAction = SafetyAction.ALLOW
    severity: SafetySeverity = SafetySeverity.NONE
    categories: list[str] = Field(default_factory=list)
    child_message: str | None = None
    parent_note: str | None = None

