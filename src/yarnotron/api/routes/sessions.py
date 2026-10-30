from fastapi import APIRouter, Depends, HTTPException, status

from yarnotron.models.operations import CreateSessionRequest, EndSessionRequest, StorySession
from yarnotron.services.session_quality import summarize
from yarnotron.services.shutdown import reject_while_draining
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "",
    response_model=StorySession,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(reject_while_draining)],
)
async def create_session(
    request: CreateSessionRequest,
    repository: StoryRepository = Depends(get_story_repository),
) -> StorySession:
    try:
        return await repository.create_session(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Story not found") from exc


@router.get("", response_model=list[StorySession])
async def list_sessions(
    repository: StoryRepository = Depends(get_story_repository),
) -> list[StorySession]:
    return await repository.list_sessions()


@router.get("/{session_id}/summary")
async def session_summary(
    session_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> dict[str, int]:
    """Counts that show how smoothly a session went (held sentences, confirmations, repairs). No words."""
    turns = [t for t in await repository.list_conversation_turns(limit=500) if t.session_id == session_id]
    return summarize(turns)


@router.post("/{session_id}/end", response_model=StorySession)
async def end_session(
    session_id: str,
    request: EndSessionRequest,
    repository: StoryRepository = Depends(get_story_repository),
) -> StorySession:
    try:
        return await repository.end_session(session_id, ended_at=request.ended_at)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
