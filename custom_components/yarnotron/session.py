"""Which backend session a child's turns belong to -- and when a grown-up must step in.

The backend counts safety strikes per session. When a session ends on its
strike limit it refuses every later turn, and its message tells the child to
"ask a grown-up to start a new one". That cool-off only works if nothing else
quietly opens a fresh session, so this manager:

* starts a session on the first turn, and a new one after the child has been
  idle for a while (a normal, unlocked restart);
* **locks** when the backend ends a session (strike limit, or a parent ended it
  on the dashboard). While locked, no turn reaches the backend. Only
  `async_start_new_session`, which is wired to a button and a service that a
  parent presses, clears it.

State is persisted so a Home Assistant restart cannot reset the lock.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .api import YarnotronApi, YarnotronApiError, YarnotronError
from .const import DOMAIN, LOCKING_CATEGORIES, STORAGE_VERSION

_LOGGER = logging.getLogger(__name__)

_SAVE_DELAY = 5.0


class SessionLockedError(Exception):
    """A grown-up has to start the next session."""


@dataclass
class SessionState:
    session_id: str | None = None
    story_id: str | None = None
    pinned_story_id: str | None = None
    locked: bool = False
    last_activity: str | None = None


class SessionManager:
    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        api: YarnotronApi,
        child_id: str,
        idle_minutes: Callable[[], int],
        on_change: Callable[[], None],
    ) -> None:
        self._api = api
        self._child_id = child_id
        self._idle_minutes = idle_minutes
        self._on_change = on_change
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}.session"
        )
        self._lock = asyncio.Lock()
        self._verified = False
        self.state = SessionState()

    # ------------------------------------------------------------- properties

    @property
    def locked(self) -> bool:
        return self.state.locked

    @property
    def session_id(self) -> str | None:
        return self.state.session_id

    @property
    def story_id(self) -> str | None:
        return self.state.story_id

    @property
    def pinned_story_id(self) -> str | None:
        return self.state.pinned_story_id

    # ---------------------------------------------------------------- storage

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if isinstance(data, dict):
            known = SessionState.__dataclass_fields__
            self.state = SessionState(**{k: v for k, v in data.items() if k in known})

    async def async_save(self) -> None:
        await self._store.async_save(asdict(self.state))

    def _save_soon(self) -> None:
        self._store.async_delay_save(lambda: asdict(self.state), _SAVE_DELAY)

    # ------------------------------------------------------------------ turns

    async def async_prepare_turn(self, device_id: str | None) -> tuple[str, str | None]:
        """The (session_id, story_id) the next turn should use.

        Raises SessionLockedError while a grown-up has yet to start a session.
        """
        async with self._lock:
            if self.state.locked:
                raise SessionLockedError

            if self.state.session_id is not None:
                if not self._verified:
                    await self._verify_stored_session()
                    if self.state.locked:
                        raise SessionLockedError
                elif self._idle_expired():
                    await self._end_quietly()

            if self.state.session_id is None:
                await self._start(device_id)

            assert self.state.session_id is not None
            return self.state.session_id, self.state.story_id

    async def async_record_turn(self, turn: dict[str, Any]) -> None:
        """Fold a finished turn back in: remember the story, notice a session ending."""
        async with self._lock:
            self.state.last_activity = dt_util.utcnow().isoformat()

            story = turn.get("updated_story")
            if isinstance(story, dict) and isinstance(story.get("id"), str):
                self.state.story_id = story["id"]

            safety = turn.get("safety") or {}
            categories = set(safety.get("categories") or [])
            if categories & LOCKING_CATEGORIES:
                self.state.locked = True
                self.state.session_id = None
                self.state.story_id = None
                await self.async_save()
                self._on_change()
                return

            self._save_soon()

    # --------------------------------------------------------- parent actions

    async def async_start_new_session(self, device_id: str | None = None) -> str:
        """A grown-up starts a fresh session. This is the only way out of a lock."""
        async with self._lock:
            if self.state.session_id is not None:
                await self._end_quietly()
            self.state.locked = False
            session_id = await self._start(device_id)
            self._on_change()
            return session_id

    async def async_end_session(self) -> None:
        """End the open session normally. Does not lock; the next turn starts a new one."""
        async with self._lock:
            await self._end_quietly()
            self._on_change()

    async def async_set_pinned_story(self, story_id: str | None) -> None:
        """Make the next session continue this story, ending the current one.

        Does not clear a lock: a session that ended on strikes still needs a
        grown-up to start the next one.
        """
        async with self._lock:
            self.state.pinned_story_id = story_id
            await self._end_quietly()
            self._on_change()

    # --------------------------------------------------------------- internals

    def _idle_expired(self) -> bool:
        if not self.state.last_activity:
            return False
        last = dt_util.parse_datetime(self.state.last_activity)
        if last is None:
            return False
        return dt_util.utcnow() - last > timedelta(minutes=self._idle_minutes())

    async def _verify_stored_session(self) -> None:
        """Check a session remembered across a restart against what the backend has.

        A session the backend has never heard of (database replaced) would run
        turns with no session at all, and strikes would stop adding up. One that
        has ended means the backend or a parent ended it while we were not
        looking, which is a lock. A newer session opened by another client just
        means ours is stale.
        """
        try:
            status = await self._api.status(self._child_id)
        except YarnotronError:
            return  # cannot tell; leave unverified and let the turn report the problem

        self._verified = True
        child = status.get("child") or {}
        latest = child.get("latest_session_id")
        if latest == self.state.session_id and child.get("latest_session_ended"):
            self.state.locked = True
            self.state.session_id = None
            self.state.story_id = None
            await self.async_save()
            self._on_change()
        elif latest != self.state.session_id:
            _LOGGER.debug("Stored session %s is stale; starting a new one", self.state.session_id)
            self.state.session_id = None
            self.state.story_id = None

    async def _end_quietly(self) -> None:
        session_id = self.state.session_id
        self.state.session_id = None
        self.state.story_id = None
        self._save_soon()
        if session_id is None:
            return
        try:
            await self._api.end_session(session_id)
        except YarnotronError as err:
            # Best effort. The next session starts regardless.
            _LOGGER.debug("Could not end session %s: %s", session_id, err)

    async def _start(self, device_id: str | None) -> str:
        story_id = self.state.pinned_story_id
        try:
            session = await self._api.create_session(self._child_id, story_id, device_id)
        except YarnotronApiError as err:
            if err.status != 404 or story_id is None:
                raise
            # The pinned story was deleted; carry on without it.
            _LOGGER.warning("Pinned story %s no longer exists; clearing it", story_id)
            self.state.pinned_story_id = None
            story_id = None
            session = await self._api.create_session(self._child_id, None, device_id)

        self.state.session_id = session["id"]
        self.state.story_id = story_id
        self.state.last_activity = dt_util.utcnow().isoformat()
        self._verified = True
        await self.async_save()
        return self.state.session_id
