from fastapi import APIRouter, Depends, HTTPException, Query

from yarnotron.agent.turn_processor import TurnProcessor, get_turn_processor
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.operations import (
    ReleaseSafetyEventRequest,
    ReviewSafetyEventRequest,
    SafetyEvent,
)
from yarnotron.models.safety import SafetyAction
from yarnotron.services.ollama import OllamaUnavailableError
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/safety-events", tags=["safety"])


@router.get("", response_model=list[SafetyEvent])
async def list_safety_events(
    limit: int = Query(default=50, ge=1, le=250),
    repository: StoryRepository = Depends(get_story_repository),
) -> list[SafetyEvent]:
    return await repository.list_safety_events(limit=limit)


@router.post("/{event_id}/review", response_model=SafetyEvent)
async def review_safety_event(
    event_id: str,
    request: ReviewSafetyEventRequest,
    repository: StoryRepository = Depends(get_story_repository),
) -> SafetyEvent:
    try:
        return await repository.review_safety_event(event_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Safety event not found") from exc


@router.post("/{event_id}/release", response_model=SafetyEvent)
async def release_safety_event(
    event_id: str,
    request: ReleaseSafetyEventRequest,
    repository: StoryRepository = Depends(get_story_repository),
    processor: TurnProcessor = Depends(get_turn_processor),
) -> SafetyEvent:
    """A parent decides the block was wrong: run the turn the child lost.

    Only a stopped turn can be released. A caution let the story continue
    already, and a session-ended refusal is not a decision about anything the
    child said, so neither has anything to give back.
    """
    event = await repository.get_safety_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Safety event not found")
    if event.safety.action not in {SafetyAction.BLOCK, SafetyAction.ALERT_PARENT}:
        raise HTTPException(status_code=409, detail="Only a blocked turn can be released")
    if "session_ended" in event.safety.categories:
        raise HTTPException(status_code=409, detail="Nothing was blocked; the session had ended")
    if event.released_at is not None:
        raise HTTPException(status_code=409, detail="This block was already released")
    if event.story_id is None:
        raise HTTPException(status_code=409, detail="This block has no story to reapply it to")

    # Recorded before the re-run, so the turn's own safety review can see that
    # a parent has allowed this line rather than blocking it a second time.
    released = await repository.release_safety_event(event_id, request)
    try:
        await processor.process(
            AgentTurnRequest(
                child_id=event.child_id,
                story_id=event.story_id,
                session_id=event.session_id,
                transcript=event.transcript_excerpt,
            ),
            parent_released=True,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return released
