"""Fixtures: a real HTTP fake backend and a config entry pointing at it."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from aiohttp.test_utils import TestServer
from custom_components.yarnotron.const import DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from .fake_backend import TOKEN, FakeBackend
from .fake_supervisor import FakeSupervisor

pytest_plugins = ["pytest_homeassistant_custom_component"]


@pytest.fixture(autouse=True)
def _enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Let Home Assistant load custom_components from this repo."""


@pytest.fixture(autouse=True)
def _allow_sockets(socket_enabled: None) -> None:
    """The fake backend is a real local HTTP server."""


@pytest.fixture(autouse=True)
async def _core(hass: HomeAssistant) -> None:
    """The conversation dependency expects Home Assistant core to be set up."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
async def backend() -> AsyncGenerator[tuple[FakeBackend, str]]:
    fake = FakeBackend()
    server = TestServer(fake.app())
    await server.start_server()
    try:
        yield fake, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


@pytest.fixture
def entry_for(hass: HomeAssistant, backend):
    def make(**options) -> MockConfigEntry:
        _, url = backend
        entry = MockConfigEntry(
            domain=DOMAIN,
            title="Yarnotron",
            data={"url": url, "api_token": TOKEN, "child_id": "local_child"},
            options=options,
            unique_id=f"{url}|local_child",
        )
        entry.add_to_hass(hass)
        return entry

    return make


@pytest.fixture
async def setup(hass: HomeAssistant, entry_for):
    async def _setup(**options) -> MockConfigEntry:
        entry = entry_for(**options)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        return entry

    return _setup


@pytest.fixture
async def supervisor() -> AsyncGenerator[tuple[FakeSupervisor, str]]:
    fake = FakeSupervisor()
    server = TestServer(fake.app())
    await server.start_server()
    try:
        yield fake, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()
