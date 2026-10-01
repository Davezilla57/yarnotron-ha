"""Buttons for a parent: start a new session, or end the current one."""

from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import YarnotronError
from .const import DOMAIN
from .entity import YarnotronEntity
from .models import YarnotronConfigEntry

PARALLEL_UPDATES = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data.coordinator
    sessions = entry.runtime_data.sessions
    async_add_entities(
        [
            StartNewSessionButton(coordinator, sessions),
            EndSessionButton(coordinator, sessions),
        ]
    )


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
