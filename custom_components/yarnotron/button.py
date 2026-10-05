"""Buttons for a parent: start a new session, or end the current one."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import YarnotronError
from .const import DOMAIN
from .entity import MacEntity, YarnotronEntity
from .models import YarnotronConfigEntry

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    sessions = entry.runtime_data.sessions
    entities: list[ButtonEntity] = [
        StartNewSessionButton(coordinator, sessions),
        EndSessionButton(coordinator, sessions),
    ]
    if entry.runtime_data.supervisor_coordinator is not None:
        entities += [
            LaunchButton(entry.runtime_data.supervisor_coordinator),
            ShutdownButton(entry.runtime_data.supervisor_coordinator),
        ]
    async_add_entities(entities)


class _SessionButton(YarnotronEntity, ButtonEntity):
    def __init__(self, coordinator, sessions, key: str) -> None:
        super().__init__(coordinator, key)
        self._sessions = sessions

    @property
    def available(self) -> bool:
        # Works from local state; a down backend is reported when pressed.
        return True


class StartNewSessionButton(_SessionButton):
    """The one way out of a lock. After the strike limit a child cannot restart alone."""

    def __init__(self, coordinator, sessions) -> None:
        super().__init__(coordinator, sessions, "start_new_session")

    async def async_press(self) -> None:
        try:
            await self._sessions.async_start_new_session()
        except YarnotronError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="request_failed",
                translation_placeholders={"error": str(err)},
            ) from err
        await self.coordinator.async_request_refresh()


class EndSessionButton(_SessionButton):
    def __init__(self, coordinator, sessions) -> None:
        super().__init__(coordinator, sessions, "end_session")

    async def async_press(self) -> None:
        await self._sessions.async_end_session()
        await self.coordinator.async_request_refresh()


class LaunchButton(MacEntity, ButtonEntity):
    """Check everything and start whatever is not answering. Returns at once; the
    outcome arrives as a notification (only when something needs attention) and
    as a `yarnotron_power_finished` event."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator, "launch")

    async def async_press(self) -> None:
        power = self.coordinator.config_entry.runtime_data.power
        power.async_launch()


class ShutdownButton(MacEntity, ButtonEntity):
    """Save the story, then stop the Mac's processes. No confirmation: a button is a grown-up's."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator, "shutdown")

    async def async_press(self) -> None:
        power = self.coordinator.config_entry.runtime_data.power
        power.async_shutdown()
