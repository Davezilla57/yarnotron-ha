from pydantic import BaseModel, Field


class RepositorySummary(BaseModel):
    """Counts straight from the datastore, scoped to one child where it matters.

    Stories belong to the household, so `stories` is global. Sessions, safety
    events and media belong to a child.
    """

    stories: int = 0
    open_sessions: int = 0
    unreviewed_safety_events: int = 0
    pending_media: int = 0
    latest_session_id: str | None = None
    latest_session_ended: bool = False
    latest_session_strikes: int = 0


class ModelsStatus(BaseModel):
    # False means Yarnotron is up but its model server is not -- worth telling a
    # client apart from "Yarnotron itself is unreachable".
    ollama_reachable: bool
    chat_model: str
    safety_model: str
    embedding_model: str | None = None
    # Configured models Ollama does not have. Empty when Ollama is unreachable:
    # unknown is not the same as everything missing.
    missing: list[str] = Field(default_factory=list)


class ChildStatus(BaseModel):
    child_id: str
    open_sessions: int
    unreviewed_safety_events: int
    pending_media: int
    latest_session_id: str | None = None
    latest_session_ended: bool = False
    latest_session_strikes: int = 0


class StatusResponse(BaseModel):
    app: str
    version: str
    environment: str
    auth_required: bool
    strike_limit: int
    stories: int
    models: ModelsStatus
    child: ChildStatus
