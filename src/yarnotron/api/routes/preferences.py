from fastapi import APIRouter, Depends, HTTPException

from yarnotron.models.preferences import (
    ParentPreferences,
    UpdateParentPreferencesRequest,
    VoiceOption,
    VoiceSampleRequest,
)
from yarnotron.services.preferences import (
    ParentPreferencesStore,
    get_parent_preferences_store,
    list_voice_options,
    play_voice_sample,
)

router = APIRouter(prefix="/preferences", tags=["preferences"])


@router.get("", response_model=ParentPreferences)
async def get_preferences(
    store: ParentPreferencesStore = Depends(get_parent_preferences_store),
) -> ParentPreferences:
    return store.get()


@router.put("", response_model=ParentPreferences)
async def update_preferences(
    request: UpdateParentPreferencesRequest,
    store: ParentPreferencesStore = Depends(get_parent_preferences_store),
) -> ParentPreferences:
    return store.update(request)


@router.get("/voices", response_model=list[VoiceOption])
async def list_voices() -> list[VoiceOption]:
    return list_voice_options()


@router.post("/voices/sample", status_code=204)
async def sample_voice(request: VoiceSampleRequest) -> None:
    try:
        play_voice_sample(
            provider=request.tts_provider,
            voice=request.tts_voice,
            text=request.text,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
