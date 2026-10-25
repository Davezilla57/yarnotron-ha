from fastapi import APIRouter, Depends, HTTPException, Response, status

from yarnotron.models.memory import MemorySearchRequest, MemorySearchResult
from yarnotron.services.memory import Embedder, embed_query, get_embedder
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=list[MemorySearchResult])
async def search_memory(
    request: MemorySearchRequest,
    embedder: Embedder = Depends(get_embedder),
    repository: StoryRepository = Depends(get_story_repository),
) -> list[MemorySearchResult]:
    query_embedding = await embed_query(embedder, request.query)
    return await repository.search_memory(
        query_embedding=query_embedding,
        story_id=request.story_id,
        limit=request.limit,
    )


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory_chunk(
    memory_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> Response:
    deleted = await repository.delete_memory_chunk(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory chunk not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
