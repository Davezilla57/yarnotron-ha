from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from yarnotron.api.routes import (
    agent,
    conversation_turns,
    dashboard,
    health,
    media,
    memory,
    models,
    preferences,
    safety,
    sessions,
    stories,
)
from yarnotron.core.config import get_settings
from yarnotron.services.ollama import OllamaClient
from yarnotron.services.repository import open_story_repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    async with open_story_repository(settings) as repository:
        app.state.story_repository = repository
        app.state.ollama_client = OllamaClient(settings)
        try:
            yield
        finally:
            await app.state.ollama_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    media_output_dir = Path(settings.media_output_dir)
    media_output_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Local-first backend for a child-safe storytelling agent.",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(dashboard.router)
    app.include_router(models.router, prefix="/v1")
    app.include_router(stories.router, prefix="/v1")
    app.include_router(agent.router, prefix="/v1")
    app.include_router(conversation_turns.router, prefix="/v1")
    app.include_router(media.router, prefix="/v1")
    app.include_router(memory.router, prefix="/v1")
    app.include_router(preferences.router, prefix="/v1")
    app.include_router(sessions.router, prefix="/v1")
    app.include_router(safety.router, prefix="/v1")
    app.mount("/media", StaticFiles(directory=media_output_dir), name="media")
    return app
