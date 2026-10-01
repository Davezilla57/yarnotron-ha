"""The Storytime switch: a parent's on/off for the whole feature."""

from __future__ import annotations

from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.const import STATE_OFF
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .entity import device_info_for
from .models import YarnotronConfigEntry

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([StorytimeSwitch(entry)])


class StorytimeSwitch(RestoreEntity, SwitchEntity):
    """While off, the agent says storytime is resting and never reaches the backend.

    Automations can drive it -- a schedule helper for quiet hours, a chore
    tracker, a presence sensor -- which is how "parent-controlled operating
    hours" from the original design is done here.
    """

    _attr_has_entity_name = True
    _attr_translation_key = "storytime"
    _attr_should_poll = False

    def __init__(self, entry: YarnotronConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_storytime"
        self._attr_device_info = device_info_for(entry)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if (last := await self.async_get_last_state()) is not None:
            self._entry.runtime_data.storytime_enabled = last.state != STATE_OFF

    @property
    def is_on(self) -> bool:
        return self._entry.runtime_data.storytime_enabled

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._entry.runtime_data.storytime_enabled = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._entry.runtime_data.storytime_enabled = False
        self.async_write_ha_state()
