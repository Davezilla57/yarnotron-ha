"""Shared entity plumbing."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .api import normalize_base_url
from .const import CONF_CHILD_ID, DEFAULT_CHILD_ID, DOMAIN
from .coordinator import YarnotronCoordinator
from .models import YarnotronConfigEntry


def device_info_for(entry: YarnotronConfigEntry) -> DeviceInfo:
    child_id = entry.data[CONF_CHILD_ID]
    name = "Yarnotron" if child_id == DEFAULT_CHILD_ID else f"Yarnotron ({child_id})"
    info = DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=name,
        manufacturer="Yarnotron",
        model="Story backend",
        entry_type=DeviceEntryType.SERVICE,
    )
    try:
        info["configuration_url"] = f"{normalize_base_url(entry.data['url'])}/dashboard"
    except (KeyError, ValueError):
        pass
    return info


class YarnotronEntity(CoordinatorEntity[YarnotronCoordinator]):
    """Base for entities that read the coordinator."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: YarnotronCoordinator, key: str) -> None:
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = device_info_for(entry)
