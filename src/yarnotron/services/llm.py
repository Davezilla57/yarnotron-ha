"""Choose the model-server client from settings.

`YARNOTRON_LLM_BACKEND=ollama` (the default) keeps Ollama's native API.
`YARNOTRON_LLM_BACKEND=openai` talks to an OpenAI-compatible server such as LiteLLM.
Both clients expose the same five methods, so nothing downstream cares which it got.
"""

from __future__ import annotations

import httpx

from yarnotron.core.config import Settings
from yarnotron.services.ollama import OllamaClient


def build_llm_client(settings: Settings, client: httpx.AsyncClient | None = None) -> OllamaClient:
    if settings.llm_backend == "openai":
        from yarnotron.services.llm_openai import OpenAICompatClient

        return OpenAICompatClient(settings, client)
    return OllamaClient(settings, client)
