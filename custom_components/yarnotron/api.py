"""Async client for the Yarnotron backend."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
from yarl import URL

from .const import REQUEST_TIMEOUT, STREAM_READ_TIMEOUT

_LOGGER = logging.getLogger(__name__)


class YarnotronError(Exception):
    """Base class for backend errors."""


class YarnotronConnectionError(YarnotronError):
    """The backend could not be reached."""


class YarnotronAuthError(YarnotronError):
    """The backend rejected the API token."""


class YarnotronApiError(YarnotronError):
    """The backend answered with an error."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def normalize_base_url(value: str) -> str:
    """Turn what a person typed into a base URL, or raise ValueError.

    "192.168.1.20:8989" gets http:// ; a trailing slash or a /dashboard path is
    dropped. Anything that is not http(s) with a host is rejected.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty URL")
    if "://" not in text:
        text = f"http://{text}"
    url = URL(text)
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError(f"not an http(s) URL: {value!r}")
    return str(url.with_path("").with_query(None).with_fragment(None)).rstrip("/")


class YarnotronApi:
    """Thin wrapper over the endpoints the integration uses."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        token: str | None = None,
    ) -> None:
        self._session = session
        self.base_url = base_url.rstrip("/")
        self._token = token or None

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    # ------------------------------------------------------------------ reads

    async def health(self) -> dict[str, Any]:
        """Reachability check. Needs no token, so it tells "down" from "wrong token"."""
        return await self._json("GET", "/health", authenticated=False)

    async def status(self, child_id: str) -> dict[str, Any]:
        return await self._json("GET", "/v1/status", params={"child_id": child_id})

    async def safety_events(self, limit: int) -> list[dict[str, Any]]:
        return await self._json("GET", "/v1/safety-events", params={"limit": limit})

    async def list_stories(self) -> list[dict[str, Any]]:
        return await self._json("GET", "/v1/stories")

    async def export_story(self, story_id: str) -> str:
        return await self._text("GET", f"/v1/stories/{story_id}/export.md")

    # ----------------------------------------------------------------- writes

    async def create_session(
        self, child_id: str, story_id: str | None, device_id: str | None
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            "/v1/sessions",
            body={"child_id": child_id, "story_id": story_id, "device_id": device_id},
        )

    async def end_session(self, session_id: str) -> dict[str, Any]:
        return await self._json("POST", f"/v1/sessions/{session_id}/end", body={})

    async def review_safety_event(
        self, event_id: str, reviewed_by: str, note: str | None
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"/v1/safety-events/{event_id}/review",
            body={"reviewed_by": reviewed_by, "review_note": note},
        )

    async def release_safety_event(
        self, event_id: str, released_by: str, note: str | None
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"/v1/safety-events/{event_id}/release",
            body={"released_by": released_by, "review_note": note},
        )

    # ------------------------------------------------------------------- turn

    async def stream_turn(
        self,
        *,
        child_id: str,
        story_id: str | None,
        session_id: str | None,
        device_id: str | None,
        transcript: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield each event of a turn as the backend produces it.

        Events are `{"sentence": ...}` while the reply is written, then one
        `{"turn": {...}}` with the full response -- or `{"error": ..., "status": ...}`.
        A non-2xx answer to the request itself (bad token, unknown route) raises.
        """
        timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=REQUEST_TIMEOUT, sock_read=STREAM_READ_TIMEOUT
        )
        try:
            async with self._session.post(
                f"{self.base_url}/v1/agent/turn/stream",
                json={
                    "child_id": child_id,
                    "story_id": story_id,
                    "session_id": session_id,
                    "device_id": device_id,
                    "transcript": transcript,
                },
                headers={**self._headers, "Accept": "application/x-ndjson"},
                timeout=timeout,
            ) as response:
                await self._raise_for_status(response)
                async for raw in response.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        _LOGGER.debug("Skipping a malformed stream line: %r", line[:80])
        except (aiohttp.ClientError, TimeoutError) as err:
            raise YarnotronConnectionError(str(err) or type(err).__name__) from err

    # -------------------------------------------------------------- internals

    async def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> Any:
        response_text = await self._text(
            method, path, params=params, body=body, authenticated=authenticated
        )
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as err:
            raise YarnotronApiError(200, f"{path} did not return JSON") from err

    async def _text(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> str:
        headers = self._headers if authenticated else {"Accept": "application/json"}
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        try:
            async with self._session.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=body,
                headers=headers,
                timeout=timeout,
            ) as response:
                await self._raise_for_status(response)
                return await response.text()
        except (aiohttp.ClientError, TimeoutError) as err:
            raise YarnotronConnectionError(str(err) or type(err).__name__) from err

    @staticmethod
    async def _raise_for_status(response: aiohttp.ClientResponse) -> None:
        if response.status < 400:
            return
        if response.status in (401, 403):
            raise YarnotronAuthError("The API token was rejected")
        detail = await response.text()
        try:
            parsed = json.loads(detail).get("detail", detail)
            detail = parsed if isinstance(parsed, str) else json.dumps(parsed)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise YarnotronApiError(response.status, detail[:300])
