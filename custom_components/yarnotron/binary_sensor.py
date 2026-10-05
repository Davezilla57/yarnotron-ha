"""Binary sensors: is the machinery up, and does a grown-up need to act?"""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .entity import MacEntity, YarnotronEntity
from .models import YarnotronConfigEntry

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    entities: list[BinarySensorEntity] = [
        OllamaReachableSensor(coordinator),
        SessionLockedSensor(coordinator, entry.runtime_data.sessions),
    ]
    if entry.runtime_data.supervisor_coordinator is not None:
        entities.append(MacSupervisorSensor(entry.runtime_data.supervisor_coordinator))
    async_add_entities(entities)


class OllamaReachableSensor(YarnotronEntity, BinarySensorEntity):
    """Yarnotron can be up while the model server behind it is not."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator, "model_server")

    @property
    def is_on(self) -> bool | None:
        reachable = self.coordinator.data.models.get("ollama_reachable")
        return None if reachable is None else bool(reachable)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        models = self.coordinator.data.models
        return {
            "chat_model": models.get("chat_model"),
            "safety_model": models.get("safety_model"),
            "embedding_model": models.get("embedding_model"),
            "missing_models": models.get("missing") or [],
        }


class SessionLockedSensor(YarnotronEntity, BinarySensorEntity):
    """On when a session ended and the child cannot continue until a grown-up acts.

    Known locally, so it stays available even while the backend is unreachable.
    """

    _attr_device_class = BinarySensorDeviceClass.PROBLEM

    def __init__(self, coordinator, sessions) -> None:
        super().__init__(coordinator, "session_locked")
        self._sessions = sessions

    @property
    def available(self) -> bool:
        return True

    @property
    def is_on(self) -> bool:
        return self._sessions.locked


class MacSupervisorSensor(MacEntity, BinarySensorEntity):
    """Whether the Mac's supervisor answers. Off means the Mac is asleep, off, or the
    supervisor is not running; the attributes say which."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator, "mac_supervisor")

    @property
    def is_on(self) -> bool:
        data = self.coordinator.data or {}
        return bool(data.get("reachable")) and data.get("problem") is None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data or {}
        return {"problem": data.get("problem"), "error": data.get("error")}
