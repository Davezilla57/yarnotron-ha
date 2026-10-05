"""Sensors: counts a parent glances at.

Attributes never carry what the child said. Entity state and attributes are
recorded in Home Assistant's database and shown on any dashboard, so the words
of a child's story stay in the backend, behind its token.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util

from .entity import MacEntity, YarnotronEntity
from .models import YarnotronConfigEntry, YarnotronData

PARALLEL_UPDATES = 0


def _latest_flagged_event(data: YarnotronData) -> dict[str, Any] | None:
    for event in data.events:  # newest first
        if (event.get("safety") or {}).get("action", "allow") != "allow":
            return event
    return None


def _last_event_time(data: YarnotronData) -> datetime | None:
    event = _latest_flagged_event(data)
    return dt_util.parse_datetime(str(event["created_at"])) if event else None


def _last_event_attrs(data: YarnotronData) -> dict[str, Any]:
    event = _latest_flagged_event(data)
    if event is None:
        return {}
    safety = event.get("safety") or {}
    return {
        "event_id": event.get("id"),
        "action": safety.get("action"),
        "severity": safety.get("severity"),
        "categories": safety.get("categories") or [],
        "parent_note": safety.get("parent_note"),
        "reviewed": event.get("reviewed_at") is not None,
    }


@dataclass(frozen=True, kw_only=True)
class YarnotronSensorDescription(SensorEntityDescription):
    value_fn: Callable[[YarnotronData], StateType | datetime]
    attrs_fn: Callable[[YarnotronData], dict[str, Any]] | None = None


SENSORS: tuple[YarnotronSensorDescription, ...] = (
    YarnotronSensorDescription(
        key="unreviewed_safety_events",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.child.get("unreviewed_safety_events"),
    ),
    YarnotronSensorDescription(
        key="session_strikes",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.child.get("latest_session_strikes", 0),
        attrs_fn=lambda data: {
            "strike_limit": data.status.get("strike_limit"),
            "session_ended": bool(data.child.get("latest_session_ended")),
        },
    ),
    YarnotronSensorDescription(
        key="last_safety_event",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=_last_event_time,
        attrs_fn=_last_event_attrs,
    ),
    YarnotronSensorDescription(
        key="pending_images",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda data: data.child.get("pending_media"),
    ),
    YarnotronSensorDescription(
        key="stories",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda data: data.status.get("stories"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    entities: list[SensorEntity] = [
        YarnotronSensor(coordinator, description) for description in SENSORS
    ]
    if entry.runtime_data.supervisor_coordinator is not None:
        entities.append(MacServicesSensor(entry.runtime_data.supervisor_coordinator))
    async_add_entities(entities)


class YarnotronSensor(YarnotronEntity, SensorEntity):
    entity_description: YarnotronSensorDescription

    def __init__(self, coordinator, description: YarnotronSensorDescription) -> None:
        super().__init__(coordinator, description.key)
        self.entity_description = description

    @property
    def native_value(self) -> StateType | datetime:
        return self.entity_description.value_fn(self.coordinator.data)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        if self.entity_description.attrs_fn is None:
            return None
        return self.entity_description.attrs_fn(self.coordinator.data)


class MacServicesSensor(MacEntity, SensorEntity):
    """One word for the Mac side, with every service in the attributes."""

    _attr_device_class = SensorDeviceClass.ENUM

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator, "mac_services")
        self._attr_options = ["all_running", "degraded", "stopped", "unreachable"]

    @property
    def native_value(self) -> str:
        data = self.coordinator.data or {}
        if not data.get("reachable") or data.get("problem"):
            return "unreachable"
        services = data.get("services", [])
        if services and all(s["ok"] for s in services):
            return "all_running"
        if services and not any(s["ok"] for s in services):
            return "stopped"
        return "degraded"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data or {}
        return {
            "services": {s["name"]: s["state"] for s in data.get("services", [])},
            "problem": data.get("problem"),
            "error": data.get("error"),
        }
