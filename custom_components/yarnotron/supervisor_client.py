"""Client for the host supervisor (see src/yarnotron/supervisor)."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from .api import YarnotronAuthError, YarnotronError

SUPERVISOR_TIMEOUT = 10


class SupervisorError(YarnotronError):
    """The supervisor answered, but not with what was expected."""


class SupervisorAuthError(YarnotronAuthError):
    """The supervisor rejected the token."""


class SupervisorUnreachableError(YarnotronError):
    """Nothing came back from the supervisor.

    `kind` says why, because the fix differs:
      refused  -- the Mac answered but nothing listens: the supervisor is not running.
      timeout  -- no answer at all: the Mac is asleep, off, or off the network.
      other    -- DNS failure, no route, TLS, anything else.
    """

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class SupervisorClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str, token: str | None) -> None:
        self._session = session
        self.base_url = base_url.rstrip("/")
        self._token = token or None

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/health", authenticated=False)

    async def status(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/status")

    async def launch(self) -> dict[str, Any]:
        return await self._request("POST", "/v1/launch")

    async def shutdown(self, *, force: bool = False) -> dict[str, Any]:
        return await self._request("POST", "/v1/shutdown", params={"force": "1"} if force else None)

    async def job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if authenticated and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            async with self._session.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=SUPERVISOR_TIMEOUT),
            ) as response:
                text = await response.text()
                if response.status == 401:
                    raise SupervisorAuthError("The supervisor token was rejected")
                if response.status >= 400:
                    raise SupervisorError(f"{response.status}: {text[:200]}")
        except aiohttp.ClientConnectorError as err:
            # "Connection refused" arrives as a ClientConnectorError wrapping ConnectionRefusedError.
            if isinstance(err.os_error, ConnectionRefusedError):
                raise SupervisorUnreachableError("refused", str(err)) from err
            raise SupervisorUnreachableError("other", str(err)) from err
        except TimeoutError as err:
            raise SupervisorUnreachableError("timeout", "no answer in time") from err
        except aiohttp.ClientError as err:
            raise SupervisorUnreachableError("other", str(err) or type(err).__name__) from err
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            raise SupervisorError(f"{path} did not return JSON") from err
