"""Polls the Host supervisor so the Mac's services show up as entities."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL, DOMAIN
from .models import YarnotronConfigEntry
from .supervisor_client import (
    SupervisorAuthError,
    SupervisorClient,
    SupervisorError,
    SupervisorUnreachableError,
)

_LOGGER = logging.getLogger(__name__)


class SupervisorCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """A Mac that is asleep is an ordinary state, not a failure.

    So an unreachable supervisor produces data (`reachable: False`) rather than
    an UpdateFailed, and the entities say "unreachable" instead of "unavailable".
    """

    config_entry: YarnotronConfigEntry

    def __init__(
        self, hass: HomeAssistant, entry: YarnotronConfigEntry, client: SupervisorClient
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} supervisor {entry.title}",
            update_interval=timedelta(
                seconds=int(entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL))
            ),
        )
        self.client = client

    async def _async_update_data(self) -> dict[str, Any]:
        try:
            status = await self.client.status()
        except SupervisorUnreachableError as err:
            return {"reachable": False, "problem": err.kind, "error": err.detail, "services": []}
        except SupervisorAuthError:
            return {"reachable": True, "problem": "auth", "error": "token rejected", "services": []}
        except SupervisorError as err:
            return {"reachable": True, "problem": "error", "error": str(err), "services": []}
        return {
            "reachable": True,
            "problem": None,
            "error": None,
            "ok": status.get("ok", False),
            "busy": status.get("busy", False),
            "services": status.get("services", []),
        }
