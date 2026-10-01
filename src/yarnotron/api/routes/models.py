from fastapi import APIRouter, Depends, HTTPException

from yarnotron.services.ollama import OllamaClient, OllamaUnavailableError, get_ollama_client

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/ollama")
async def list_ollama_models(
    ollama: OllamaClient = Depends(get_ollama_client),
) -> dict[str, object]:
    try:
        return {"models": await ollama.list_models()}
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

