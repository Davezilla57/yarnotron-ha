"""Config flow: connect to a Yarnotron backend and name the child it serves."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .api import (
    YarnotronApi,
    YarnotronApiError,
    YarnotronAuthError,
    YarnotronConnectionError,
    YarnotronError,
    normalize_base_url,
)
from .const import (
    CONF_API_TOKEN,
    CONF_CHILD_ID,
    CONF_IDLE_MINUTES,
    CONF_INCLUDE_TRANSCRIPTS,
    CONF_KEEP_LISTENING,
    CONF_SCAN_INTERVAL,
    DEFAULT_CHILD_ID,
    DEFAULT_IDLE_MINUTES,
    DEFAULT_INCLUDE_TRANSCRIPTS,
    DEFAULT_KEEP_LISTENING,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    MAX_SCAN_INTERVAL,
    MIN_SCAN_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)

CONF_URL = "url"

_TOKEN_SELECTOR = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))


async def _validate(hass, url: str, token: str | None, child_id: str) -> tuple[str, dict[str, str]]:
    """Return (normalised url, errors). Empty errors means the backend is usable."""
    try:
        base = normalize_base_url(url)
    except ValueError:
        return url, {"base": "invalid_url"}
    api = YarnotronApi(async_get_clientsession(hass), base, token)
    try:
        await api.health()
        await api.status(child_id)
    except YarnotronAuthError:
        return base, {"base": "invalid_auth"}
    except YarnotronConnectionError:
        return base, {"base": "cannot_connect"}
    except YarnotronApiError as err:
        # An older backend without /v1/status answers 404: it is Yarnotron, but
        # not the version this integration needs.
        return base, {"base": "unsupported_backend" if err.status == 404 else "unknown"}
    except YarnotronError:
        return base, {"base": "unknown"}
    except Exception:
        _LOGGER.exception("Unexpected error validating Yarnotron backend")
        return base, {"base": "unknown"}
    return base, {}


def _user_schema(defaults: Mapping[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_URL, default=defaults.get(CONF_URL, "")): str,
            vol.Optional(
                CONF_API_TOKEN, description={"suggested_value": defaults.get(CONF_API_TOKEN)}
            ): _TOKEN_SELECTOR,
            vol.Required(CONF_CHILD_ID, default=defaults.get(CONF_CHILD_ID, DEFAULT_CHILD_ID)): str,
        }
    )


class YarnotronConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            child_id = user_input[CONF_CHILD_ID].strip() or DEFAULT_CHILD_ID
            token = (user_input.get(CONF_API_TOKEN) or "").strip() or None
            base, errors = await _validate(self.hass, user_input[CONF_URL], token, child_id)
            if not errors:
                await self.async_set_unique_id(f"{base}|{child_id}")
                self._abort_if_unique_id_configured()
                data = {CONF_URL: base, CONF_CHILD_ID: child_id}
                if token:
                    data[CONF_API_TOKEN] = token
                return self.async_create_entry(
                    title="Yarnotron"
                    if child_id == DEFAULT_CHILD_ID
                    else f"Yarnotron ({child_id})",
                    data=data,
                )
        return self.async_show_form(
            step_id="user", data_schema=_user_schema(user_input or {}), errors=errors
        )

    async def async_step_reauth(self, entry_data: Mapping[str, Any]) -> ConfigFlowResult:
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            token = (user_input.get(CONF_API_TOKEN) or "").strip() or None
            _, errors = await _validate(
                self.hass, entry.data[CONF_URL], token, entry.data[CONF_CHILD_ID]
            )
            if not errors:
                data = {k: v for k, v in entry.data.items() if k != CONF_API_TOKEN}
                if token:
                    data[CONF_API_TOKEN] = token
                return self.async_update_reload_and_abort(entry, data=data)
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema({vol.Optional(CONF_API_TOKEN): _TOKEN_SELECTOR}),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the backend address or token; the child stays the same."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            token = (user_input.get(CONF_API_TOKEN) or "").strip() or None
            base, errors = await _validate(
                self.hass, user_input[CONF_URL], token, entry.data[CONF_CHILD_ID]
            )
            if not errors:
                await self.async_set_unique_id(f"{base}|{entry.data[CONF_CHILD_ID]}")
                self._abort_if_unique_id_mismatch(reason="wrong_child")
                data = {CONF_URL: base, CONF_CHILD_ID: entry.data[CONF_CHILD_ID]}
                if token:
                    data[CONF_API_TOKEN] = token
                return self.async_update_reload_and_abort(entry, data=data)
        defaults = user_input or entry.data
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_URL, default=defaults.get(CONF_URL, "")): str,
                    vol.Optional(
                        CONF_API_TOKEN,
                        description={"suggested_value": defaults.get(CONF_API_TOKEN)},
                    ): _TOKEN_SELECTOR,
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry) -> YarnotronOptionsFlow:
        return YarnotronOptionsFlow()


class YarnotronOptionsFlow(OptionsFlowWithReload):
    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        o = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_SCAN_INTERVAL, default=o.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=MIN_SCAN_INTERVAL,
                        max=MAX_SCAN_INTERVAL,
                        step=5,
                        unit_of_measurement="s",
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_IDLE_MINUTES, default=o.get(CONF_IDLE_MINUTES, DEFAULT_IDLE_MINUTES)
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=720,
                        step=1,
                        unit_of_measurement="min",
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(
                    CONF_KEEP_LISTENING, default=o.get(CONF_KEEP_LISTENING, DEFAULT_KEEP_LISTENING)
                ): BooleanSelector(),
                vol.Required(
                    CONF_INCLUDE_TRANSCRIPTS,
                    default=o.get(CONF_INCLUDE_TRANSCRIPTS, DEFAULT_INCLUDE_TRANSCRIPTS),
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
