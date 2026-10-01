"""The backend still speaks what custom_components/yarnotron reads.

The integration is tested against a fake backend (ha_tests/fake_backend.py).
A fake can drift from the real thing, so this pins the field names the
integration relies on to the backend's real models. If one is renamed here,
update the integration and its fake with it.
"""

from __future__ import annotations

from pathlib import Path

from yarnotron.models.operations import (
    ReleaseSafetyEventRequest,
    ReviewSafetyEventRequest,
    SafetyEvent,
)
from yarnotron.models.safety import SafetyDecision
from yarnotron.models.status import ChildStatus, ModelsStatus, StatusResponse


def test_status_fields_the_integration_reads() -> None:
    assert {"strike_limit", "stories", "models", "child"} <= set(StatusResponse.model_fields)
    assert {
        "ollama_reachable",
        "chat_model",
        "safety_model",
        "embedding_model",
        "missing",
    } <= set(ModelsStatus.model_fields)
    assert {
        "unreviewed_safety_events",
        "pending_media",
        "latest_session_id",
        "latest_session_ended",
        "latest_session_strikes",
    } <= set(ChildStatus.model_fields)


def test_safety_event_fields_the_integration_reads() -> None:
    assert {
        "id",
        "child_id",
        "story_id",
        "session_id",
        "safety",
        "transcript_excerpt",
        "counted_as_strike",
        "reviewed_at",
        "created_at",
    } <= set(SafetyEvent.model_fields)
    assert {"action", "severity", "categories", "parent_note"} <= set(SafetyDecision.model_fields)


def test_review_and_release_bodies_match() -> None:
    assert {"reviewed_by", "review_note"} <= set(ReviewSafetyEventRequest.model_fields)
    assert {"released_by", "review_note"} <= set(ReleaseSafetyEventRequest.model_fields)


def test_session_ending_categories_are_still_emitted() -> None:
    """custom_components/yarnotron/const.py LOCKING_CATEGORIES must be real categories."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "yarnotron" / "agent" / "turn_processor.py"
    ).read_text()
    for category in ("strike_limit_reached", "session_ended"):
        assert f'"{category}"' in source, f"the backend no longer emits {category!r}"
