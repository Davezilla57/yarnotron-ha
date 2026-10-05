"""Diagnostics download, with the token and the child's words left out."""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from .const import CONF_API_TOKEN, CONF_MAC_ADDRESS, CONF_SUPERVISOR_TOKEN, CONF_SUPERVISOR_URL
from .models import YarnotronConfigEntry

TO_REDACT = {
    CONF_API_TOKEN,
    CONF_SUPERVISOR_TOKEN,
    CONF_SUPERVISOR_URL,
    CONF_MAC_ADDRESS,
    "url", "transcript", "transcript_excerpt", "parent_note",
}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: YarnotronConfigEntry
) -> dict[str, Any]:
    runtime = entry.runtime_data
    data = runtime.coordinator.data
    return {
        "entry": async_redact_data(dict(entry.data), TO_REDACT),
        "options": async_redact_data(dict(entry.options), TO_REDACT),
        "status": async_redact_data(data.status, TO_REDACT) if data else None,
        "recent_event_count": len(data.events) if data else 0,
        "session": {
            "has_session": runtime.sessions.session_id is not None,
            "locked": runtime.sessions.locked,
        },
        "storytime_enabled": runtime.storytime_enabled,
        "powered_down": runtime.powered_down,
        "mac": (
            async_redact_data(runtime.supervisor_coordinator.data or {}, TO_REDACT)
            if runtime.supervisor_coordinator
            else None
        ),
    }
