from __future__ import annotations

from datetime import timedelta

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


async def _poll(hass: HomeAssistant, seconds: int = 0) -> None:
    """Run one poll now.

    Interval timers run as background tasks that block_till_done does not wait
    for, so tests refresh explicitly. test_polls_on_the_configured_interval
    checks the schedule itself.
    """
    (entry,) = hass.config_entries.async_entries("yarnotron")
    await entry.runtime_data.coordinator.async_refresh()
    await hass.async_block_till_done()


async def test_setup_and_unload(hass: HomeAssistant, setup) -> None:
    entry = await setup()
    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get("sensor.yarnotron_stories").state == "1"
    assert hass.states.get("binary_sensor.yarnotron_model_server").state == "on"
    assert hass.states.get("binary_sensor.yarnotron_session_locked").state == "off"
    assert hass.states.get("switch.yarnotron_storytime").state == "on"

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_backend_down_at_boot_still_sets_up(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    fake.down = True
    entry = await setup()
    assert entry.state is ConfigEntryState.LOADED
    assert hass.states.get("sensor.yarnotron_stories").state == "unavailable"
    # The lock is local knowledge, so it stays readable while the backend is away.
    assert hass.states.get("binary_sensor.yarnotron_session_locked").state == "off"

    fake.down = False
    await _poll(hass)
    assert hass.states.get("sensor.yarnotron_stories").state == "1"


async def test_bad_token_starts_reauth(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    fake.token = "other"
    entry = await setup()
    assert entry.state is ConfigEntryState.SETUP_ERROR or entry.state is ConfigEntryState.LOADED
    flows = hass.config_entries.flow.async_progress_by_handler("yarnotron")
    assert any(f["context"]["source"] == "reauth" for f in flows)


async def test_old_events_are_history_not_news(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    fake.add_event("old", created_at=dt_util.utcnow() - timedelta(hours=1))
    await setup()
    assert hass.states.get("event.yarnotron_safety_alert").state == "unknown"


async def test_new_event_alerts_once(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup()
    fake.add_event("e1", action="block", categories=["violence"])
    await _poll(hass)
    state = hass.states.get("event.yarnotron_safety_alert")
    assert state.attributes["event_type"] == "blocked"
    assert state.attributes["event_id"] == "e1"
    first = state.state

    await _poll(hass, 70)  # same event, polled again: no second alert
    assert hass.states.get("event.yarnotron_safety_alert").state == first


async def test_allowed_turns_do_not_alert(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup()
    fake.add_event("fine", action="allow")
    await _poll(hass)
    assert hass.states.get("event.yarnotron_safety_alert").state == "unknown"


async def test_session_ending_event_type(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup()
    fake.add_event("end", action="block", categories=["strike_limit_reached"])
    await _poll(hass)
    assert (
        hass.states.get("event.yarnotron_safety_alert").attributes["event_type"] == "session_ended"
    )


async def test_other_childs_events_are_ignored(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup()
    fake.add_event("sibling", child_id="someone_else")
    await _poll(hass)
    assert hass.states.get("event.yarnotron_safety_alert").state == "unknown"


async def test_transcript_only_when_opted_in(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup()
    fake.add_event("e1", transcript="the child said this")
    await _poll(hass)
    assert "transcript" not in hass.states.get("event.yarnotron_safety_alert").attributes
    assert "the child said this" not in str(hass.states.get("sensor.yarnotron_last_safety_event"))


async def test_transcript_included_when_enabled(hass: HomeAssistant, backend, setup) -> None:
    fake, _ = backend
    await setup(include_transcripts=True)
    fake.add_event("e1", transcript="the child said this")
    await _poll(hass)
    assert (
        hass.states.get("event.yarnotron_safety_alert").attributes["transcript"]
        == "the child said this"
    )


async def test_last_safety_event_sensor_has_no_transcript(
    hass: HomeAssistant, backend, setup
) -> None:
    fake, _ = backend
    fake.add_event("e0", categories=["violence"], transcript="private words")
    await setup()
    state = hass.states.get("sensor.yarnotron_last_safety_event")
    assert state.attributes["categories"] == ["violence"]
    assert "private words" not in str(state.attributes)


async def test_polls_on_the_configured_interval(hass: HomeAssistant, setup) -> None:
    entry = await setup(scan_interval=90)
    assert entry.runtime_data.coordinator.update_interval == timedelta(seconds=90)
