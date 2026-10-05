"""Draining the backend before it is stopped.

Every repository call opens its own connection and commits, so a finished turn
is already saved. What can still be in flight when someone asks to shut down:

* a turn that is mid-stream (the child is hearing the reply),
* deferred input reviews and interest lookups, which run as background tasks
  after the reply and write safety events or interests when they finish.

Stopping the process while either is running loses that work, and a lost safety
review is the worst thing to lose. This module counts both, so a shutdown can
refuse new turns, wait for what is running, and say what it could not finish.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import HTTPException, status

SHUTTING_DOWN_DETAIL = "Yarnotron is shutting down and is not taking new turns."


@dataclass
class DrainReport:
    drained: bool
    waited_seconds: float
    turns_in_flight_at_start: int
    turns_still_running: int
    background_tasks_finished: int
    background_tasks_cancelled: int


class ShutdownCoordinator:
    def __init__(self) -> None:
        self.draining = False
        self._turns = 0
        self._tasks: set[asyncio.Task] = set()

    @property
    def turns_in_flight(self) -> int:
        return self._turns

    @property
    def background_tasks(self) -> int:
        return sum(1 for task in self._tasks if not task.done())

    @contextlib.contextmanager
    def turn(self) -> Iterator[None]:
        """Count a turn as running for as long as the `with` block is."""
        self._turns += 1
        try:
            yield
        finally:
            self._turns -= 1

    def track_task(self, task: asyncio.Task) -> None:
        """Remember a background task so a shutdown can wait for it."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def begin(self) -> None:
        self.draining = True

    def cancel(self) -> None:
        self.draining = False

    async def drain(self, timeout: float) -> DrainReport:
        """Wait up to `timeout` seconds for running work, then cancel what is left."""
        started = time.monotonic()
        turns_at_start = self._turns
        pending_at_start = {task for task in self._tasks if not task.done()}

        while time.monotonic() - started < timeout:
            if self._turns == 0 and self.background_tasks == 0:
                break
            await asyncio.sleep(0.05)

        leftover = [task for task in self._tasks if not task.done()]
        for task in leftover:
            task.cancel()
        if leftover:
            await asyncio.gather(*leftover, return_exceptions=True)

        return DrainReport(
            drained=self._turns == 0 and not leftover,
            waited_seconds=round(time.monotonic() - started, 2),
            turns_in_flight_at_start=turns_at_start,
            turns_still_running=self._turns,
            background_tasks_finished=len(pending_at_start) - len(leftover),
            background_tasks_cancelled=len(leftover),
        )


_COORDINATOR = ShutdownCoordinator()


def get_shutdown() -> ShutdownCoordinator:
    return _COORDINATOR


def reset_shutdown() -> ShutdownCoordinator:
    """A fresh coordinator. For tests; the app keeps one for its whole life."""
    global _COORDINATOR
    _COORDINATOR = ShutdownCoordinator()
    return _COORDINATOR


def reject_while_draining() -> None:
    """Route dependency: refuse new work once a shutdown has begun."""
    if _COORDINATOR.draining:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, SHUTTING_DOWN_DETAIL)
