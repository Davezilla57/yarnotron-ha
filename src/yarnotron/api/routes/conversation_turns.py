from fastapi import APIRouter, Depends, Query

from yarnotron.models.memory import ConversationTurn
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/conversation-turns", tags=["conversation-turns"])


@router.get("", response_model=list[ConversationTurn])
async def list_conversation_turns(
    limit: int = Query(default=20, ge=1, le=100),
    repository: StoryRepository = Depends(get_story_repository),
) -> list[ConversationTurn]:
    return await repository.list_conversation_turns(limit=limit)
