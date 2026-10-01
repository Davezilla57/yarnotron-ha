from fastapi import APIRouter, Depends, HTTPException, Query, status

from yarnotron.agent.safety import SafetyReviewer, get_safety_reviewer
from yarnotron.models.media import (
    CreateMediaAssetRequest,
    GenerateMediaAssetRequest,
    MediaAsset,
    MediaAssetStatus,
    ReviewMediaAssetRequest,
)
from yarnotron.models.safety import SafetyAction
from yarnotron.services.media_generator import MediaGenerator, get_media_generator
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(prefix="/media", tags=["media"])


@router.get("", response_model=list[MediaAsset])
async def list_media_assets(
    limit: int = Query(default=50, ge=1, le=250),
    repository: StoryRepository = Depends(get_story_repository),
) -> list[MediaAsset]:
    return await repository.list_media_assets(limit=limit)


@router.post("/requests", response_model=MediaAsset, status_code=status.HTTP_201_CREATED)
async def create_media_request(
    request: CreateMediaAssetRequest,
    repository: StoryRepository = Depends(get_story_repository),
    safety: SafetyReviewer = Depends(get_safety_reviewer),
) -> MediaAsset:
    decision = await safety.review_transcript(f"Generated image request: {request.prompt}")
    status_to_save = (
        MediaAssetStatus.PENDING_PARENT_REVIEW
        if decision.action == SafetyAction.ALLOW
        else MediaAssetStatus.REJECTED
    )
    return await repository.create_media_asset(request, safety=decision, status=status_to_save)


@router.post("/{media_id}/approve", response_model=MediaAsset)
async def approve_media_asset(
    media_id: str,
    request: ReviewMediaAssetRequest,
    repository: StoryRepository = Depends(get_story_repository),
) -> MediaAsset:
    asset = await repository.get_media_asset(media_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Media asset not found")
    if asset.status != MediaAssetStatus.PENDING_PARENT_REVIEW:
        raise HTTPException(status_code=409, detail="Only pending media can be approved")
    try:
        return await repository.approve_media_asset(media_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Media asset not found") from exc


@router.post("/{media_id}/reject", response_model=MediaAsset)
async def reject_media_asset(
    media_id: str,
    request: ReviewMediaAssetRequest,
    repository: StoryRepository = Depends(get_story_repository),
) -> MediaAsset:
    try:
        return await repository.reject_media_asset(media_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Media asset not found") from exc


@router.post("/{media_id}/generate", response_model=MediaAsset)
async def generate_media_asset(
    media_id: str,
    request: GenerateMediaAssetRequest,
    repository: StoryRepository = Depends(get_story_repository),
    safety: SafetyReviewer = Depends(get_safety_reviewer),
    generator: MediaGenerator = Depends(get_media_generator),
) -> MediaAsset:
    asset = await repository.get_media_asset(media_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Media asset not found")
    if asset.status != MediaAssetStatus.APPROVED:
        raise HTTPException(status_code=409, detail="Media must be approved before generation")
    if request.generator != "local_svg_placeholder":
        raise HTTPException(status_code=400, detail="Unknown media generator")

    decision = await safety.review_transcript(f"Approved generated image prompt: {asset.prompt}")
    if decision.action != SafetyAction.ALLOW:
        failed = await repository.mark_media_asset_failed(
            media_id,
            "Safety review blocked the approved prompt during generation.",
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Media prompt failed the final safety gate",
                "media_asset": failed.model_dump(mode="json"),
            },
        )

    try:
        generated = await generator.generate(asset)
    except Exception as exc:
        failed = await repository.mark_media_asset_failed(media_id, str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Media generation failed",
                "media_asset": failed.model_dump(mode="json"),
            },
        ) from exc

    return await repository.mark_media_asset_generated(
        media_id,
        asset_path=generated.asset_path,
        asset_url=generated.asset_url,
        mime_type=generated.mime_type,
        generator=generated.generator,
        revised_prompt=generated.revised_prompt,
    )
