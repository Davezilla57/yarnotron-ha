import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def application_support_dir() -> Path:
    """Where a user's Yarnotron data lives, outside the source tree.

    An installed app cannot write into the repository it was built from, and a
    child's stories should survive deleting the checkout. Honours
    YARNOTRON_DATA_DIR, then XDG_DATA_HOME, then the platform convention.

    macOS keeps ~/Library/Application Support so existing installs do not move.
    Linux (including containers) uses the XDG default; it used to fall through
    to a ~/Library path that does not exist there.
    """
    override = os.environ.get("YARNOTRON_DATA_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "Yarnotron"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Yarnotron"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Yarnotron"
    return Path.home() / ".local" / "share" / "Yarnotron"


class Settings(BaseSettings):
    app_name: str = "Yarnotron"
    environment: str = "local"
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "gemma4:12b-mlx"
    # Never None. OllamaClient resolves None to default_model, so leaving it
    # unset silently ran every safety review on the 12B story model -- which is
    # slower and, on a bare sentence, worse: it blocked "she deflates the toys
    # sometimes when she bites them" as sexual content in a story about a dog.
    # e2b is what .env.example and phase-0-baseline both already recommend.
    safety_model: str = "gemma4:e2b-mlx"
    embedding_model: str | None = "nomic-embed-text:v1.5"
    embedding_dimensions: int = Field(default=768, ge=1)
    request_timeout_seconds: float = Field(default=30, ge=1)
    # Which model server to talk to. "ollama" speaks Ollama's native API (the
    # default, unchanged). "openai" speaks the OpenAI-compatible API, which is
    # what LiteLLM, vLLM, llama.cpp and most local engines serve; with it the
    # *_model settings below are alias or model names on that server and
    # `llm_base_url` (including /v1) is required.
    llm_backend: Literal["ollama", "openai"] = "ollama"
    # "classic" is the turn pipeline; "agent" is the Talker/Scribe engine (see agent/agent_engine.py).
    engine: Literal["classic", "agent"] = "classic"
    agent_scribe_budget_seconds: float = Field(default=6.0, gt=0)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    # Ask the server for a JSON object (response_format). Off by default: the
    # reply is parsed tolerantly either way, and not every engine supports it.
    llm_json_mode: bool = False
    # Send chat_template_kwargs.enable_thinking so servers that support it turn
    # reasoning off, as Ollama's `think: false` does. Turn off for a strict
    # OpenAI endpoint that rejects unknown request fields.
    llm_thinking_hint: bool = True
    # Some embedding models want different prefixes for a search query and for a
    # stored passage (EmbeddingGemma: "task: search result | query:" and
    # "title: none | text:"). Empty means no prefix, which is right for
    # nomic-embed-text and the hash fallback.
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    # temperature:0 alone does not make generation repeatable -- two identical
    # corpus runs disagreed on 25% of patch decisions. Setting a seed pins it,
    # which is what the golden corpus needs to be able to detect a regression.
    # Leave unset in normal use so replies vary between sessions.
    model_seed: int | None = None
    # How many cautioned turns a child gets in one session before it ends.
    # Only the milder keyword tier counts toward this; sexual content and
    # self-harm still stop the turn on the first mention.
    safety_strike_limit: int = Field(default=5, ge=1)
    # A file, not a server.
    database_url: str = str(application_support_dir() / "yarnotron.db")
    media_output_dir: str = str(application_support_dir() / "generated_media")
    preferences_path: str = str(application_support_dir() / "preferences.json")
    # When set, every /v1 route requires `Authorization: Bearer <token>`.
    # Unset (or blank) keeps the old trust-the-network behaviour, which is right
    # for a laptop and wrong for anything reachable from a home LAN.
    api_token: str | None = None
    # A username and password for people. The dashboard asks for them with the browser's own sign-in prompt, and
    # they open the /v1 routes too, so the dashboard never needs the API token. Unset keeps the dashboard as before.
    dashboard_user: str = "parent"
    dashboard_password: str | None = None

    @field_validator("api_token", "dashboard_password", "llm_api_key", "llm_base_url")
    @classmethod
    def _blank_token_means_none(cls, value: str | None) -> str | None:
        # `YARNOTRON_API_TOKEN=` in a .env file must not lock everyone out.
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def _openai_backend_needs_a_url(self) -> "Settings":
        if self.llm_backend == "openai" and not self.llm_base_url:
            raise ValueError("YARNOTRON_LLM_BASE_URL is required when YARNOTRON_LLM_BACKEND=openai")
        return self

    model_config = SettingsConfigDict(
        env_prefix="YARNOTRON_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
