"""Runtime data shared by the integration's platforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry

if TYPE_CHECKING:
    from .api import YarnotronApi
    from .coordinator import YarnotronCoordinator
    from .power import PowerController
    from .session import SessionManager
    from .supervisor_client import SupervisorClient
    from .supervisor_coordinator import SupervisorCoordinator


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
    # The Mac supervisor, when its address is set in the options.
    supervisor: SupervisorClient | None = None
    supervisor_coordinator: SupervisorCoordinator | None = None
    power: PowerController | None = None
    mac_address: str | None = None
    # True after a successful shutdown, until the next launch. While set the
    # agent says so instead of trying a backend that is deliberately off.
    powered_down: bool = False
    # Monotonic deadline for a spoken "shut down" that is waiting for a "yes".
    shutdown_confirm_until: float = 0.0


type YarnotronConfigEntry = ConfigEntry[YarnotronRuntimeData]
