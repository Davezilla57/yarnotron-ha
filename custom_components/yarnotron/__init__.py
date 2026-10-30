"""Yarnotron: a child-safe storytelling agent for Home Assistant Assist."""

from __future__ import annotations

import logging

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import ConfigType

from .api import YarnotronApi, normalize_base_url
from .const import (
    CONF_API_TOKEN,
    CONF_CHILD_ID,
    CONF_FINISHED_SPEAKING_ENTITIES,
    CONF_FINISHED_SPEAKING_MODE,
    CONF_IDLE_MINUTES,
    CONF_MAC_ADDRESS,
    CONF_SUPERVISOR_TOKEN,
    CONF_SUPERVISOR_URL,
    DEFAULT_FINISHED_SPEAKING_MODE,
    DEFAULT_IDLE_MINUTES,
    DOMAIN,
)
from .coordinator import YarnotronCoordinator
from .models import YarnotronConfigEntry, YarnotronRuntimeData
from .power import PowerController
from .services import async_setup_services
from .session import SessionManager
from .vad import async_apply_finished_speaking
from .supervisor_client import SupervisorClient
from .supervisor_coordinator import SupervisorCoordinator

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.BUTTON,
    Platform.CONVERSATION,
    Platform.EVENT,
    Platform.SENSOR,
    Platform.SWITCH,
]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: YarnotronConfigEntry) -> bool:
    api = YarnotronApi(
        async_get_clientsession(hass),
        normalize_base_url(entry.data["url"]),
        entry.data.get(CONF_API_TOKEN),
    )
    coordinator = YarnotronCoordinator(hass, entry, api)

    def _changed() -> None:
        coordinator.async_update_listeners()

    sessions = SessionManager(
        hass,
        entry.entry_id,
        api,
        entry.data[CONF_CHILD_ID],
        lambda: int(entry.options.get(CONF_IDLE_MINUTES, DEFAULT_IDLE_MINUTES)),
        _changed,
    )
    await coordinator.async_load()
    await sessions.async_load()

    # Not async_config_entry_first_refresh: the backend is often on another
    # machine that may boot after Home Assistant. The agent should still exist
    # and apologise aloud; entities show as unavailable until it appears.
    # An authentication failure still starts re-authentication.
    await coordinator.async_refresh()

    runtime = YarnotronRuntimeData(api=api, coordinator=coordinator, sessions=sessions)
    entry.runtime_data = runtime
    runtime.power = PowerController(hass, entry)

    supervisor_url = (entry.options.get(CONF_SUPERVISOR_URL) or "").strip()
    if supervisor_url:
        runtime.supervisor = SupervisorClient(
            async_get_clientsession(hass),
            normalize_base_url(supervisor_url),
            entry.options.get(CONF_SUPERVISOR_TOKEN),
        )
        runtime.mac_address = (entry.options.get(CONF_MAC_ADDRESS) or "").strip() or None
        runtime.supervisor_coordinator = SupervisorCoordinator(hass, entry, runtime.supervisor)
        await runtime.supervisor_coordinator.async_refresh()

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    waiting = list(entry.options.get(CONF_FINISHED_SPEAKING_ENTITIES) or [])
    if waiting:
        mode = entry.options.get(CONF_FINISHED_SPEAKING_MODE, DEFAULT_FINISHED_SPEAKING_MODE)

        async def _apply(_hass: HomeAssistant) -> None:
            await async_apply_finished_speaking(hass, waiting, mode)

        # The satellites' selects may not exist yet while Home Assistant is still starting.
        entry.async_on_unload(async_at_started(hass, _apply))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: YarnotronConfigEntry) -> bool:
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.sessions.async_save()
        await entry.runtime_data.coordinator.async_shutdown()
    return unloaded
