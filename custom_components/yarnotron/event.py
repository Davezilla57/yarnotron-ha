"""The safety-alert event entity: what automations trigger on.

Each new flagged turn raises one event. Use it as a state trigger on
`event.yarnotron_safety_alert` (see blueprints/) to push a notification to a
parent's phone, which is the part the original Yarnotron only flagged
(`parent_alert_sent`) and never actually delivered.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components.event import EventEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    ALERT_BLOCKED,
    ALERT_CAUTION,
    ALERT_SESSION_ENDED,
    ALERT_TYPES,
    CONF_INCLUDE_TRANSCRIPTS,
    DEFAULT_INCLUDE_TRANSCRIPTS,
    LOCKING_CATEGORIES,
)
from .entity import YarnotronEntity
from .models import YarnotronConfigEntry

PARALLEL_UPDATES = 0

_BLOCKING_ACTIONS = {"block", "alert_parent"}


def alert_type(event: dict[str, Any]) -> str:
    """Classify a backend safety event for automations."""
    safety = event.get("safety") or {}
    if set(safety.get("categories") or []) & LOCKING_CATEGORIES:
        return ALERT_SESSION_ENDED
    if safety.get("action") in _BLOCKING_ACTIONS:
        return ALERT_BLOCKED
    return ALERT_CAUTION


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([SafetyAlertEvent(entry.runtime_data.coordinator, entry)])


class SafetyAlertEvent(YarnotronEntity, EventEntity):
    _attr_event_types = ALERT_TYPES

    def __init__(self, coordinator, entry: YarnotronConfigEntry) -> None:
        super().__init__(coordinator, "safety_alert")
        self._entry = entry

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(self.coordinator.async_add_alert_listener(self._handle_alert))

    @callback
    def _handle_alert(self, event: dict[str, Any]) -> None:
        safety = event.get("safety") or {}
        attributes: dict[str, Any] = {
            "event_id": event.get("id"),
            "action": safety.get("action"),
            "severity": safety.get("severity"),
            "categories": safety.get("categories") or [],
            "parent_note": safety.get("parent_note"),
            "story_id": event.get("story_id"),
            "session_id": event.get("session_id"),
            "counted_as_strike": bool(event.get("counted_as_strike")),
            "created_at": event.get("created_at"),
        }
        # Off by default: this attribute is recorded in Home Assistant's
        # database and visible on any dashboard that shows the entity.
        if self._entry.options.get(CONF_INCLUDE_TRANSCRIPTS, DEFAULT_INCLUDE_TRANSCRIPTS):
            attributes["transcript"] = event.get("transcript_excerpt")
        self._trigger_event(alert_type(event), attributes)
        self.async_write_ha_state()

    @property
    def available(self) -> bool:
        # Alerts are pushed from polling; a missed poll does not make the
        # entity itself unavailable, and its history should stay readable.
        return True
