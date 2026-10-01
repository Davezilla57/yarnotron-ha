"""Polls the backend and turns new safety events into alerts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .api import YarnotronApi, YarnotronAuthError, YarnotronError
from .const import (
    CONF_CHILD_ID,
    CONF_SCAN_INTERVAL,
    DEFAULT_SCAN_INTERVAL,
    DEFERRED_REVIEW_DELAY,
    DOMAIN,
    SAFETY_EVENT_WINDOW,
    STORAGE_VERSION,
)
from .models import YarnotronConfigEntry, YarnotronData

_LOGGER = logging.getLogger(__name__)

_REMEMBERED_ALERTS = 200

AlertListener = Callable[[dict[str, Any]], None]


class YarnotronCoordinator(DataUpdateCoordinator[YarnotronData]):
    """Backend status, plus de-duplicated safety alerts.

    The backend is the source of truth for safety events. Alerts are therefore
    derived from polling it rather than from the turn that caused them: some
    events are written a moment after the child hears a reply (the model's
    review runs behind the reply), and polling catches both kinds the same way.
    """

    config_entry: YarnotronConfigEntry

    def __init__(self, hass: HomeAssistant, entry: YarnotronConfigEntry, api: YarnotronApi) -> None:
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} {entry.title}",
            update_interval=timedelta(
                seconds=int(entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL))
            ),
        )
        self.api = api
        self.child_id: str = entry.data[CONF_CHILD_ID]
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.alerts"
        )
        # Events from before this moment are history, not news.
        self._since: datetime = dt_util.utcnow()
        self._alerted: list[str] = []
        self._alert_listeners: list[AlertListener] = []
        self._cancel_followup: CALLBACK_TYPE | None = None

    # ---------------------------------------------------------------- storage

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if not isinstance(data, dict):
            return
        since = dt_util.parse_datetime(data.get("since", ""))
        if since is not None:
            self._since = since
        ids = data.get("alerted")
        if isinstance(ids, list):
            self._alerted = [str(item) for item in ids][-_REMEMBERED_ALERTS:]

    def _save_soon(self) -> None:
        self._store.async_delay_save(
            lambda: {"since": self._since.isoformat(), "alerted": self._alerted}, 5.0
        )

    # ----------------------------------------------------------------- alerts

    @callback
    def async_add_alert_listener(self, listener: AlertListener) -> CALLBACK_TYPE:
        self._alert_listeners.append(listener)

        @callback
        def remove() -> None:
            if listener in self._alert_listeners:
                self._alert_listeners.remove(listener)

        return remove

    @callback
    def async_schedule_followup(self) -> None:
        """Poll once more shortly: the backend reviews a turn just after replying."""
        if self._cancel_followup is not None:
            self._cancel_followup()

        async def _followup(_now: datetime) -> None:
            self._cancel_followup = None
            await self.async_request_refresh()

        self._cancel_followup = async_call_later(
            self.hass, DEFERRED_REVIEW_DELAY.total_seconds(), _followup
        )

    async def async_shutdown(self) -> None:
        if self._cancel_followup is not None:
            self._cancel_followup()
            self._cancel_followup = None
        await super().async_shutdown()

    @callback
    def _dispatch_alerts(self, events: list[dict[str, Any]]) -> None:
        fresh: list[dict[str, Any]] = []
        for event in events:
            if event.get("id") in self._alerted:
                continue
            created = dt_util.parse_datetime(str(event.get("created_at", "")))
            if created is None or created < self._since:
                continue
            if (event.get("safety") or {}).get("action", "allow") == "allow":
                # A note for the parent dashboard, not something to wake anyone for.
                continue
            fresh.append(event)

        for event in sorted(fresh, key=lambda item: str(item.get("created_at"))):
            self._alerted.append(event["id"])
            for listener in list(self._alert_listeners):
                listener(event)
        if fresh:
            del self._alerted[:-_REMEMBERED_ALERTS]
            self._save_soon()

    # ------------------------------------------------------------------- poll

    async def _async_update_data(self) -> YarnotronData:
        try:
            status = await self.api.status(self.child_id)
            events = await self.api.safety_events(SAFETY_EVENT_WINDOW)
        except YarnotronAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except YarnotronError as err:
            raise UpdateFailed(f"Could not reach Yarnotron: {err}") from err

        mine = [event for event in events if event.get("child_id") == self.child_id]
        self._dispatch_alerts(mine)
        return YarnotronData(status=status, events=mine)
