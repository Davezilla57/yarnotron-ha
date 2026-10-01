import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    model_config = SettingsConfigDict(
        env_prefix="YARNOTRON_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
