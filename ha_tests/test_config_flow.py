from __future__ import annotations

from custom_components.yarnotron.const import DOMAIN
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .fake_backend import TOKEN


async def _start(hass):
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )


async def test_user_flow_creates_entry(hass: HomeAssistant, backend) -> None:
    _, url = backend
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": url, "api_token": TOKEN, "child_id": "local_child"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {"url": url, "api_token": TOKEN, "child_id": "local_child"}


async def test_bare_host_gets_http_scheme(hass: HomeAssistant, backend) -> None:
    _, url = backend
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": url.removeprefix("http://"), "api_token": TOKEN, "child_id": "x"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["url"] == url


async def test_wrong_token(hass: HomeAssistant, backend) -> None:
    _, url = backend
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": url, "api_token": "nope", "child_id": "local_child"}
    )
    assert result["errors"] == {"base": "invalid_auth"}


async def test_unreachable(hass: HomeAssistant, backend) -> None:
    fake, url = backend
    fake.down = True
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": url, "api_token": TOKEN, "child_id": "local_child"}
    )
    assert result["errors"] == {"base": "unknown"} or result["errors"] == {"base": "cannot_connect"}


async def test_old_backend_without_status(hass: HomeAssistant, backend) -> None:
    fake, url = backend
    fake.status_404 = True
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": url, "api_token": TOKEN, "child_id": "local_child"}
    )
    assert result["errors"] == {"base": "unsupported_backend"}


async def test_invalid_url(hass: HomeAssistant) -> None:
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": "ftp://x", "api_token": "", "child_id": "c"}
    )
    assert result["errors"] == {"base": "invalid_url"}


async def test_duplicate_is_aborted(hass: HomeAssistant, backend, entry_for) -> None:
    _, url = backend
    entry_for()
    flow = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        flow["flow_id"], {"url": url, "api_token": TOKEN, "child_id": "local_child"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_updates_token(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    entry = await setup()
    fake.token = "rotated"
    result = await entry.start_reauth_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"api_token": "rotated"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert entry.data["api_token"] == "rotated"


async def test_options_flow(hass: HomeAssistant, setup) -> None:
    entry = await setup()
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "scan_interval": 60,
            "idle_minutes": 10,
            "keep_listening": False,
            "include_transcripts": True,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["include_transcripts"] is True
    assert isinstance(entry, MockConfigEntry)
