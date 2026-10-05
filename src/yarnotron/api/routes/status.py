from fastapi import APIRouter, Depends, Query, Request

from yarnotron.core.config import Settings, get_settings
from yarnotron.models.status import ChildStatus, ModelsStatus, StatusResponse
from yarnotron.services.ollama import OllamaClient, OllamaUnavailableError, get_ollama_client
from yarnotron.services.shutdown import get_shutdown
from yarnotron.services.story_store import StoryRepository, get_story_repository

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
async def get_status(
    request: Request,
    child_id: str = Query(default="local_child", min_length=1),
    repository: StoryRepository = Depends(get_story_repository),
    ollama: OllamaClient = Depends(get_ollama_client),
    settings: Settings = Depends(get_settings),
) -> StatusResponse:
    """What a polling client needs, in one cheap call.

    Always answers 200 while Yarnotron itself is up. A model server that is
    down is reported in `models.ollama_reachable`, not as an error, so a client
    can say "Yarnotron is fine, Ollama is not" instead of "something broke".
    """
    summary = await repository.status_summary(child_id)
    models = await _models_status(ollama, settings)
    return StatusResponse(
        app=settings.app_name,
        version=request.app.version,
        environment=settings.environment,
        auth_required=settings.api_token is not None,
        strike_limit=settings.safety_strike_limit,
        shutting_down=get_shutdown().draining,
        stories=summary.stories,
        models=models,
        child=ChildStatus(
            child_id=child_id,
            open_sessions=summary.open_sessions,
            unreviewed_safety_events=summary.unreviewed_safety_events,
            pending_media=summary.pending_media,
            latest_session_id=summary.latest_session_id,
            latest_session_ended=summary.latest_session_ended,
            latest_session_strikes=summary.latest_session_strikes,
        ),
    )


async def _models_status(ollama: OllamaClient, settings: Settings) -> ModelsStatus:
    configured = [settings.default_model, settings.safety_model]
    if settings.embedding_model:
        configured.append(settings.embedding_model)

    try:
        installed = {model.name for model in await ollama.list_models()}
    except OllamaUnavailableError:
        return ModelsStatus(
            ollama_reachable=False,
            chat_model=settings.default_model,
            safety_model=settings.safety_model,
            embedding_model=settings.embedding_model,
        )

    return ModelsStatus(
        ollama_reachable=True,
        chat_model=settings.default_model,
        safety_model=settings.safety_model,
        embedding_model=settings.embedding_model,
        missing=[name for name in dict.fromkeys(configured) if name not in installed],
    )
