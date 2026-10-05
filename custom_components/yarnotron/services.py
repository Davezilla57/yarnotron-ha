"""Parent-facing services.

Registered once for the integration. Each takes a `config_entry_id` so more
than one child (one backend entry per child) can be addressed; when only one
entry is loaded it may be omitted.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .api import YarnotronApiError, YarnotronError
from .const import DOMAIN
from .models import YarnotronConfigEntry

SERVICE_REVIEW = "review_safety_event"
SERVICE_RELEASE = "release_safety_event"
SERVICE_START_NEW_SESSION = "start_new_session"
SERVICE_END_SESSION = "end_session"
SERVICE_SET_STORY = "set_story"
SERVICE_LIST_STORIES = "list_stories"
SERVICE_EXPORT_STORY = "export_story"
SERVICE_LAUNCH = "launch"
SERVICE_SHUTDOWN = "shutdown"
SERVICE_STATUS = "status"

_ENTRY = {vol.Optional("config_entry_id"): cv.string}

SCHEMAS: dict[str, vol.Schema] = {
    SERVICE_REVIEW: vol.Schema(
        {**_ENTRY, vol.Required("event_id"): cv.string, vol.Optional("note"): cv.string}
    ),
    SERVICE_RELEASE: vol.Schema(
        {**_ENTRY, vol.Required("event_id"): cv.string, vol.Optional("note"): cv.string}
    ),
    SERVICE_START_NEW_SESSION: vol.Schema(_ENTRY),
    SERVICE_END_SESSION: vol.Schema(_ENTRY),
    SERVICE_SET_STORY: vol.Schema({**_ENTRY, vol.Optional("story_id"): vol.Any(None, cv.string)}),
    SERVICE_LIST_STORIES: vol.Schema(_ENTRY),
    SERVICE_EXPORT_STORY: vol.Schema({**_ENTRY, vol.Required("story_id"): cv.string}),
    SERVICE_LAUNCH: vol.Schema({**_ENTRY, vol.Optional("wait", default=True): cv.boolean}),
    SERVICE_SHUTDOWN: vol.Schema(
        {
            **_ENTRY,
            vol.Optional("force", default=False): cv.boolean,
            vol.Optional("wait", default=True): cv.boolean,
        }
    ),
    SERVICE_STATUS: vol.Schema(_ENTRY),
}


def _entry_for(hass: HomeAssistant, call: ServiceCall) -> YarnotronConfigEntry:
    loaded = [
        e for e in hass.config_entries.async_entries(DOMAIN) if e.state is ConfigEntryState.LOADED
    ]
    wanted = call.data.get("config_entry_id")
    if wanted:
        for entry in loaded:
            if entry.entry_id == wanted:
                return entry
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_found",
            translation_placeholders={"entry": wanted},
        )
    if len(loaded) == 1:
        return loaded[0]
    raise ServiceValidationError(
        translation_domain=DOMAIN,
        translation_key="entry_ambiguous" if loaded else "entry_not_loaded",
    )


async def _actor(hass: HomeAssistant, call: ServiceCall) -> str:
    """Who is acting, for the audit trail the backend keeps."""
    user_id = call.context.user_id
    if user_id and (user := await hass.auth.async_get_user(user_id)) is not None:
        return f"ha:{user.name}"
    return "ha:automation"


def _failed(err: YarnotronError) -> HomeAssistantError:
    if isinstance(err, YarnotronApiError) and err.status == 404:
        return ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="not_found",
            translation_placeholders={"error": str(err)},
        )
    return HomeAssistantError(
        translation_domain=DOMAIN,
        translation_key="request_failed",
        translation_placeholders={"error": str(err)},
    )


def async_setup_services(hass: HomeAssistant) -> None:
    async def review(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        try:
            await entry.runtime_data.api.review_safety_event(
                call.data["event_id"], await _actor(hass, call), call.data.get("note")
            )
        except YarnotronError as err:
            raise _failed(err) from err
        await entry.runtime_data.coordinator.async_request_refresh()
        return None

    async def release(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        try:
            await entry.runtime_data.api.release_safety_event(
                call.data["event_id"], await _actor(hass, call), call.data.get("note")
            )
        except YarnotronError as err:
            raise _failed(err) from err
        await entry.runtime_data.coordinator.async_request_refresh()
        return None

    async def start_new_session(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        try:
            await entry.runtime_data.sessions.async_start_new_session()
        except YarnotronError as err:
            raise _failed(err) from err
        await entry.runtime_data.coordinator.async_request_refresh()
        return None

    async def end_session(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        await entry.runtime_data.sessions.async_end_session()
        await entry.runtime_data.coordinator.async_request_refresh()
        return None

    async def set_story(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        await entry.runtime_data.sessions.async_set_pinned_story(call.data.get("story_id") or None)
        return None

    async def list_stories(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        try:
            stories = await entry.runtime_data.api.list_stories()
        except YarnotronError as err:
            raise _failed(err) from err
        return {"stories": stories}

    async def export_story(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        try:
            markdown = await entry.runtime_data.api.export_story(call.data["story_id"])
        except YarnotronError as err:
            raise _failed(err) from err
        return {"story_id": call.data["story_id"], "markdown": markdown}

    async def _power(call: ServiceCall, operation) -> ServiceResponse:
        if not call.data["wait"]:
            return {"started": True, "kind": operation.kind}
        result = await operation.wait(float("inf"))
        return {
            "ok": result.ok,
            "kind": result.kind,
            "summary": result.spoken,
            "details": result.markdown,
        }

    async def launch(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        return await _power(call, entry.runtime_data.power.async_launch())

    async def shutdown(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        return await _power(
            call, entry.runtime_data.power.async_shutdown(force=call.data["force"])
        )

    async def status(call: ServiceCall) -> ServiceResponse:
        entry = _entry_for(hass, call)
        report = await entry.runtime_data.power.async_status()
        return {
            "ok": report.ok,
            "summary": report.spoken(),
            "components": report.as_dict(),
            "markdown": report.markdown(),
        }

    handlers: dict[str, Any] = {
        SERVICE_REVIEW: (review, SupportsResponse.NONE),
        SERVICE_RELEASE: (release, SupportsResponse.NONE),
        SERVICE_START_NEW_SESSION: (start_new_session, SupportsResponse.NONE),
        SERVICE_END_SESSION: (end_session, SupportsResponse.NONE),
        SERVICE_SET_STORY: (set_story, SupportsResponse.NONE),
        SERVICE_LIST_STORIES: (list_stories, SupportsResponse.ONLY),
        SERVICE_EXPORT_STORY: (export_story, SupportsResponse.ONLY),
        SERVICE_LAUNCH: (launch, SupportsResponse.OPTIONAL),
        SERVICE_SHUTDOWN: (shutdown, SupportsResponse.OPTIONAL),
        SERVICE_STATUS: (status, SupportsResponse.ONLY),
    }
    for name, (handler, response) in handlers.items():
        hass.services.async_register(
            DOMAIN, name, handler, schema=SCHEMAS[name], supports_response=response
        )
