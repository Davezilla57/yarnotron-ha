from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import PlainTextResponse

from yarnotron.models.operations import RestoreStoryVersionResponse, StoryVersion
from yarnotron.models.story import CreateStoryRequest, Story, StoryPatch
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/stories", tags=["stories"])


@router.post("", response_model=Story, status_code=status.HTTP_201_CREATED)
async def create_story(
    request: CreateStoryRequest,
    repository: StoryRepository = Depends(get_story_repository),
) -> Story:
    return await repository.create(request)


@router.get("", response_model=list[Story])
async def list_stories(
    repository: StoryRepository = Depends(get_story_repository),
) -> list[Story]:
    return await repository.list()


@router.get("/{story_id}", response_model=Story)
async def get_story(
    story_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> Story:
    story = await repository.get(story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return story


@router.delete("/{story_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_story(
    story_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> Response:
    deleted = await repository.delete(story_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Story not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{story_id}/export.md", response_class=PlainTextResponse)
async def export_story_markdown(
    story_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> PlainTextResponse:
    story = await repository.get(story_id)
    if story is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return PlainTextResponse(_story_to_markdown(story), media_type="text/markdown")


@router.post("/{story_id}/patches", response_model=Story)
async def apply_story_patch(
    story_id: str,
    patch: StoryPatch,
    repository: StoryRepository = Depends(get_story_repository),
) -> Story:
    try:
        return await repository.apply_patch(story_id, patch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Story not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{story_id}/versions", response_model=list[StoryVersion])
async def list_story_versions(
    story_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> list[StoryVersion]:
    return await repository.list_versions(story_id)


@router.post("/{story_id}/versions/{version_id}/restore", response_model=RestoreStoryVersionResponse)
async def restore_story_version(
    story_id: str,
    version_id: str,
    repository: StoryRepository = Depends(get_story_repository),
) -> RestoreStoryVersionResponse:
    try:
        return await repository.restore_version(story_id, version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Story or version not found") from exc


def _story_to_markdown(story: Story) -> str:
    lines = [
        f"# {story.title}",
        "",
        f"- Story ID: `{story.id}`",
        f"- Created: {story.created_at.isoformat()}",
        f"- Updated: {story.updated_at.isoformat()}",
        "",
    ]
    if story.synopsis:
        lines.extend(["## Synopsis", "", story.synopsis, ""])

    if story.characters:
        lines.extend(["## Characters", ""])
        for character in story.characters:
            details = []
            if character.role:
                details.append(character.role)
            if character.traits:
                details.append(", ".join(character.traits))
            suffix = f" - {'; '.join(details)}" if details else ""
            lines.append(f"- **{character.name}**{suffix}")
            if character.notes:
                lines.append(f"  - {character.notes}")
        lines.append("")

    if story.scenes:
        lines.extend(["## Scenes", ""])
        for scene in sorted(story.scenes, key=lambda item: item.sequence):
            lines.extend([f"### {scene.sequence}. {scene.title}", "", scene.summary, ""])
            if scene.draft_text:
                lines.extend([scene.draft_text, ""])

    if story.facts:
        lines.extend(["## Story Facts", ""])
        lines.extend(f"- {fact.text}" for fact in story.facts)
        lines.append("")

    return "\n".join(lines).strip() + "\n"
