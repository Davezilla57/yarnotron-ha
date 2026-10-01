"""Runtime data shared by the integration's platforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry

if TYPE_CHECKING:
    from .api import YarnotronApi
    from .coordinator import YarnotronCoordinator
    from .session import SessionManager


@dataclass
class YarnotronData:
    """One poll of the backend."""

    status: dict[str, Any]
    # The child's recent safety events, newest first.
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def child(self) -> dict[str, Any]:
        return self.status.get("child") or {}

    @property
    def models(self) -> dict[str, Any]:
        return self.status.get("models") or {}


@dataclass
class YarnotronRuntimeData:
    api: YarnotronApi
    coordinator: YarnotronCoordinator
    sessions: SessionManager
    # Driven by the "Storytime" switch. While False the agent answers with a
    # fixed message and never reaches the backend.
    storytime_enabled: bool = True
    # What the last reply said, for "repeat that". Kept in memory only: it is
    # the child's story, so it is not written to disk or to entity attributes.
    last_response: str | None = None


type YarnotronConfigEntry = ConfigEntry[YarnotronRuntimeData]
