from __future__ import annotations

import pytest
from custom_components.yarnotron.const import DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError


async def test_review_records_who_and_note(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup()
    await hass.services.async_call(
        DOMAIN, "review_safety_event", {"event_id": "e1", "note": "fine"}, blocking=True
    )
    event_id, body = fake.reviewed[0]
    assert event_id == "e1"
    assert body["review_note"] == "fine"
    assert body["reviewed_by"] == "ha:automation"  # no user in an automation context


async def test_release_unknown_event_is_a_validation_error(hass, backend, setup) -> None:
    await setup()
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "release_safety_event", {"event_id": "gone"}, blocking=True
        )


async def test_backend_failure_is_a_clean_error(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    fake.down = True
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            DOMAIN, "review_safety_event", {"event_id": "e1"}, blocking=True
        )


async def test_list_and_export_return_responses(hass, backend, setup) -> None:
    await setup()
    listed = await hass.services.async_call(
        DOMAIN, "list_stories", {}, blocking=True, return_response=True
    )
    assert listed["stories"][0]["id"] == "story_1"
    exported = await hass.services.async_call(
        DOMAIN, "export_story", {"story_id": "story_1"}, blocking=True, return_response=True
    )
    assert exported["markdown"].startswith("# story_1")


async def test_unknown_entry_id(hass, backend, setup) -> None:
    await setup()
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "end_session", {"config_entry_id": "nope"}, blocking=True
        )


async def test_buttons(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.yarnotron_start_new_session"}, blocking=True
    )
    assert fake.sessions_created == 1
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.yarnotron_end_session"}, blocking=True
    )
    assert fake.ended == ["sess_1"]


async def test_strike_sensor_and_pending_counts(hass, backend, setup) -> None:
    fake, _ = backend
    fake.strikes = 3
    fake.add_event("e1", action="redirect")
    await setup()
    assert hass.states.get("sensor.yarnotron_session_strikes").state == "3"
    assert hass.states.get("sensor.yarnotron_session_strikes").attributes["strike_limit"] == 5
    assert hass.states.get("sensor.yarnotron_unreviewed_safety_events").state == "1"


async def test_model_server_down_flag(hass, backend, setup) -> None:
    fake, _ = backend
    fake.ollama = False
    await setup()
    assert hass.states.get("binary_sensor.yarnotron_model_server").state == "off"


async def test_storytime_switch_restores_off(hass, backend, setup) -> None:
    entry = await setup()
    await hass.services.async_call(
        "switch", "turn_off", {"entity_id": "switch.yarnotron_storytime"}, blocking=True
    )
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("switch.yarnotron_storytime").state == "off"
    assert entry.runtime_data.storytime_enabled is False


async def test_diagnostics_redact_secrets(hass, hass_client, backend, setup) -> None:
    from custom_components.yarnotron.diagnostics import async_get_config_entry_diagnostics
    from homeassistant.components.diagnostics import REDACTED

    fake, _ = backend
    fake.add_event("e1", transcript="private words")
    entry = await setup()
    data = await async_get_config_entry_diagnostics(hass, entry)
    assert data["entry"]["api_token"] == REDACTED
    assert data["entry"]["url"] == REDACTED
    assert "private words" not in str(data)
