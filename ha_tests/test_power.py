"""Launch, shut down and status: by voice, by button, by service."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, patch

import pytest
from custom_components.yarnotron import power as power_module
from custom_components.yarnotron.const import (
    EVENT_POWER_FINISHED,
    POWER_DISABLED_TEXT,
    POWERED_DOWN_TEXT,
    SHUTDOWN_CANCELLED_TEXT,
    SHUTDOWN_CONFIRM_TEXT,
)
from custom_components.yarnotron.controls import ControlIntent, classify, is_confirmation
from custom_components.yarnotron.wol import magic_packet, parse_mac
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er

from .fake_supervisor import SUP_TOKEN

AGENT = "conversation.yarnotron"


@pytest.fixture(autouse=True)
def _fast_polling():
    with (
        patch.object(power_module, "JOB_POLL_SECONDS", 0.01),
        patch.object(power_module, "WAKE_POLL_SECONDS", 0.01),
    ):
        yield


async def _say(hass: HomeAssistant, text: str):
    result = await conversation.async_converse(
        hass, text, None, Context(), language="en", agent_id=AGENT
    )
    return result, result.response.speech["plain"]["speech"]


@pytest.fixture
def powered(setup, supervisor):
    _, url = supervisor

    async def _setup(**options):
        options.setdefault("supervisor_url", url)
        options.setdefault("supervisor_token", SUP_TOKEN)
        return await setup(**options)

    return _setup


def add_wyoming(hass: HomeAssistant, *, piper: bool = True, whisper: bool = True) -> None:
    registry = er.async_get(hass)
    for domain, key, name, up in (
        ("tts", "piper", "piper", piper),
        ("stt", "whisper", "faster-whisper", whisper),
    ):
        entry = registry.async_get_or_create(
            domain, "wyoming", key, suggested_object_id=name, original_name=name.title()
        )
        hass.states.async_set(entry.entity_id, "unavailable" if not up else "unknown")


# ----------------------------------------------------------------- phrases


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Launch Yarnotron", ControlIntent.LAUNCH),
        ("please start yarnotron", ControlIntent.LAUNCH),
        ("Yarnotron, launch Yarnotron", ControlIntent.LAUNCH),
        ("Shut down Yarnotron.", ControlIntent.SHUTDOWN),
        ("turn off yarnotron please", ControlIntent.SHUTDOWN),
        ("Yarnotron status", ControlIntent.STATUS),
        ("is yarnotron running?", ControlIntent.STATUS),
        ("status", ControlIntent.STATUS),
        ("goodbye", ControlIntent.STOP_SESSION),
        ("the dragon starts the engine", None),
        ("launch the rocket", None),
        ("shut down the castle", None),
    ],
)
def test_power_phrases_match_whole_utterances_only(text, intent) -> None:
    assert classify(text) is intent


def test_confirmation_words() -> None:
    assert is_confirmation("yes, shut down") is True
    assert is_confirmation("shut down yarnotron") is True
    assert is_confirmation("no") is False
    assert is_confirmation("tell me about a fox") is None


def test_magic_packet_shape() -> None:
    assert parse_mac("AA-bb-cc-dd-ee-ff") == bytes.fromhex("aabbccddeeff")
    packet = magic_packet(parse_mac("aa:bb:cc:dd:ee:ff"))
    assert len(packet) == 102 and packet[:6] == b"\xff" * 6
    with pytest.raises(ValueError, match="MAC"):
        parse_mac("nonsense")


# ------------------------------------------------------------------ status


async def test_status_lists_both_sides_and_is_calm_when_all_is_well(hass, powered) -> None:
    add_wyoming(hass)
    await powered()
    _, speech = await _say(hass, "Yarnotron status")
    assert "On Home Assistant:" in speech and "On the Mac:" in speech
    for name in ("Yarnotron voice agent", "Piper", "Ollama", "Yarnotron backend"):
        assert name in speech
    assert "Everything is working." in speech


async def test_status_explains_what_is_wrong_on_both_sides(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    fake.states["whisper"] = "unresponsive"
    fake.states["ollama"] = "down"
    add_wyoming(hass, piper=False)
    await powered()
    with patch("homeassistant.components.persistent_notification.async_create") as notify:
        _, speech = await _say(hass, "yarnotron status")

    assert "Whisper speech-to-text not responding" in speech
    assert "Ollama not running" in speech
    assert "need a look" in speech and "notifications" in speech
    body = notify.call_args.args[1]
    assert "can't reach Piper" in body  # Home Assistant's side
    assert "Port held but not answering." in body  # the Mac's side, with the observed detail
    assert "/Users/x/.yarnotron-supervisor/logs/whisper.log" in body


async def test_status_works_without_a_supervisor(hass, setup) -> None:
    await setup()
    _, speech = await _say(hass, "yarnotron status")
    assert "Mac supervisor unknown" in speech


async def test_status_distinguishes_a_stopped_supervisor_from_a_sleeping_mac(hass, setup) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
    await setup(supervisor_url=f"http://127.0.0.1:{closed_port}")  # nothing listens: refused
    with patch("homeassistant.components.persistent_notification.async_create") as notify:
        _, speech = await _say(hass, "status")
    assert "On the Mac: Mac supervisor not running" in speech
    assert "supervisor isn't running" in notify.call_args.args[1]
    assert "launchctl kickstart" in notify.call_args.args[1]


async def test_status_works_while_storytime_is_off(hass, powered) -> None:
    await powered()
    await hass.services.async_call(
        "switch", "turn_off", {"entity_id": "switch.yarnotron_storytime"}, blocking=True
    )
    _, speech = await _say(hass, "yarnotron status")
    assert "On the Mac:" in speech


# ------------------------------------------------------------------ launch


async def test_launch_starts_what_is_down_and_reports_it(
    hass, backend, supervisor, powered
) -> None:
    fake_backend, _ = backend
    fake, _ = supervisor
    fake.states.update(whisper="down", yarnotron="down")
    fake_backend.shutting_down = True  # left draining by an earlier shutdown
    await powered()
    events = []
    hass.bus.async_listen(EVENT_POWER_FINISHED, events.append)

    result, speech = await _say(hass, "launch yarnotron")

    assert "Started" in speech and "Everything on the Mac is running." in speech
    assert fake.states == dict.fromkeys(fake.states, "up")
    assert fake_backend.cancel_calls == 1  # it takes stories again
    assert result.continue_conversation is False
    await hass.async_block_till_done()
    assert events and events[0].data["ok"] is True


async def test_launch_with_everything_up_changes_nothing(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    await powered()
    _, speech = await _say(hass, "start yarnotron")
    assert speech.startswith("Everything on the Mac was already running.")
    assert fake.launch_calls == 1


async def test_launch_failure_speaks_the_cause_and_keeps_the_detail(
    hass, supervisor, powered
) -> None:
    fake, _ = supervisor
    fake.states.update(whisper="down")
    fake.launch_fails = "whisper"
    await powered()
    events = []
    hass.bus.async_listen(EVENT_POWER_FINISHED, events.append)

    with patch("homeassistant.components.persistent_notification.async_create") as notify:
        _, speech = await _say(hass, "launch yarnotron")
        await hass.async_block_till_done()

    assert "started and then quit straight away (exit code 3)" in speech
    assert "notifications" in speech
    body = notify.call_args.args[1]
    assert "boom: model file missing" in body  # the log tail
    assert "Run the command in Terminal" in body  # what to try
    assert events[0].data["ok"] is False


async def test_launch_wakes_a_sleeping_mac_first(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    fake.drop_requests = 2  # the supervisor's Mac doesn't answer until woken
    await powered(mac_address="aa:bb:cc:dd:ee:ff")
    fake.drop_requests = 2
    with patch.object(power_module, "send_magic_packet", AsyncMock()) as wake:
        _, speech = await _say(hass, "launch yarnotron")
    wake.assert_awaited_once_with("aa:bb:cc:dd:ee:ff")
    assert speech.startswith("I woke the Mac.")


async def test_launch_explains_a_mac_that_never_wakes(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    await powered(mac_address="aa:bb:cc:dd:ee:ff")
    fake.drop_requests = 10_000
    with (
        patch.object(power_module, "send_magic_packet", AsyncMock()),
        patch.object(power_module, "WAKE_WAIT_SECONDS", 0.05),
        patch("homeassistant.components.persistent_notification.async_create") as notify,
    ):
        _, speech = await _say(hass, "launch yarnotron")
    assert "I sent a wake-up signal" in speech
    assert "System Settings > Battery" in notify.call_args.args[1]


async def test_launch_says_when_the_token_is_wrong(hass, supervisor, powered) -> None:
    await powered(supervisor_token="wrong")
    _, speech = await _say(hass, "launch yarnotron")
    assert "rejected my token" in speech


# ---------------------------------------------------------------- shutdown


async def test_shutdown_asks_first_then_saves_and_stops(hass, backend, supervisor, powered) -> None:
    fake_backend, _ = backend
    fake, _ = supervisor
    entry = await powered()

    result, speech = await _say(hass, "shut down yarnotron")
    assert speech == SHUTDOWN_CONFIRM_TEXT
    assert result.continue_conversation is True
    assert fake.shutdown_calls == []  # nothing yet

    _, speech = await _say(hass, "yes, shut down")
    assert speech.startswith("Saved The Brave Fox (4 scenes).")
    assert fake.shutdown_calls == [{}]
    assert entry.runtime_data.powered_down is True

    # A child who then talks to it is told, kindly, and the backend is left alone.
    _, speech = await _say(hass, "tell me a story")
    assert speech == POWERED_DOWN_TEXT
    assert fake_backend.turns == []


async def test_shutdown_ends_the_open_session_first(hass, backend, supervisor, powered) -> None:
    fake_backend, _ = backend
    await powered()
    await _say(hass, "tell me a story")  # opens sess_1
    await _say(hass, "shut down yarnotron")
    await _say(hass, "yes shut down")
    assert "sess_1" in fake_backend.ended


async def test_shutdown_can_be_declined(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    await powered()
    await _say(hass, "shut down yarnotron")
    _, speech = await _say(hass, "no")
    assert speech == SHUTDOWN_CANCELLED_TEXT
    assert fake.shutdown_calls == []


async def test_an_unrelated_reply_cancels_a_pending_shutdown(
    hass, backend, supervisor, powered
) -> None:
    fake_backend, _ = backend
    fake, _ = supervisor
    entry = await powered()
    await _say(hass, "shut down yarnotron")
    await _say(hass, "tell me about a fox")  # carries on with the story instead
    await _say(hass, "yes")
    assert fake.shutdown_calls == []
    assert len(fake_backend.turns) == 2  # "yes" was just part of the story
    assert entry.runtime_data.shutdown_confirm_until == 0.0


async def test_an_old_confirmation_request_is_forgotten(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    entry = await powered()
    await _say(hass, "shut down yarnotron")
    entry.runtime_data.shutdown_confirm_until = 1.0  # long past
    await _say(hass, "yes shut down")
    assert fake.shutdown_calls == []


async def test_failed_shutdown_leaves_things_running_and_says_why(
    hass, supervisor, powered
) -> None:
    fake, _ = supervisor
    fake.shutdown_fails = True
    entry = await powered()
    await _say(hass, "shut down yarnotron")
    with patch("homeassistant.components.persistent_notification.async_create") as notify:
        _, speech = await _say(hass, "yes shut down")
        await hass.async_block_till_done()
    assert "refused the save-and-drain request (HTTP 500)" in speech
    assert entry.runtime_data.powered_down is False
    assert "forced shutdown" in notify.call_args.args[1] or "force" in notify.call_args.args[1]


async def test_shutdown_service_can_force(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    fake.shutdown_fails = True
    await powered()
    response = await hass.services.async_call(
        "yarnotron", "shutdown", {"force": True}, blocking=True, return_response=True
    )
    assert response["ok"] is True
    assert fake.shutdown_calls == [{"force": "1"}]


async def test_shutdown_with_the_supervisor_out_of_reach_still_saves(
    hass, backend, supervisor, powered
) -> None:
    fake_backend, _ = backend
    fake, _ = supervisor
    await powered()
    fake.drop_requests = 10_000
    await _say(hass, "shut down yarnotron")
    with patch("homeassistant.components.persistent_notification.async_create") as notify:
        _, speech = await _say(hass, "yes shut down")
    assert fake_backend.prepare_calls == 1
    assert "I saved The Brave Fox" in speech
    assert "can't reach the Mac supervisor" in speech
    assert "could not be stopped" in notify.call_args.args[1]


async def test_voice_power_can_be_turned_off(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    await powered(voice_power=False)
    _, speech = await _say(hass, "launch yarnotron")
    assert speech == POWER_DISABLED_TEXT
    _, speech = await _say(hass, "shut down yarnotron")
    assert speech == POWER_DISABLED_TEXT
    assert fake.launch_calls == 0 and fake.shutdown_calls == []
    _, speech = await _say(hass, "yarnotron status")  # looking is always allowed
    assert "On the Mac:" in speech


# -------------------------------------------------- entities and services


async def test_mac_entities_exist_only_with_a_supervisor(hass, setup, powered) -> None:
    await setup()
    registry = er.async_get(hass)
    assert registry.async_get_entity_id("button", "yarnotron", "launch") is None
    assert not any("mac_services" in e.unique_id for e in registry.entities.values())


async def test_mac_sensor_and_supervisor_sensor_follow_the_mac(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    entry = await powered()
    assert hass.states.get("sensor.yarnotron_mac_services").state == "all_running"
    assert hass.states.get("binary_sensor.yarnotron_mac_supervisor").state == "on"

    fake.states["whisper"] = "down"
    await entry.runtime_data.supervisor_coordinator.async_refresh()
    state = hass.states.get("sensor.yarnotron_mac_services")
    assert state.state == "degraded" and state.attributes["services"]["whisper"] == "down"

    fake.drop_requests = 10_000
    await entry.runtime_data.supervisor_coordinator.async_refresh()
    assert hass.states.get("sensor.yarnotron_mac_services").state == "unreachable"
    assert hass.states.get("binary_sensor.yarnotron_mac_supervisor").state == "off"


async def test_launch_button_starts_things(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    fake.states["ollama"] = "down"
    entry = await powered()
    events = []
    hass.bus.async_listen(EVENT_POWER_FINISHED, events.append)
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.yarnotron_launch_yarnotron"}, blocking=True
    )
    # Pressing returns at once; the work carries on in the background.
    await entry.runtime_data.power.last_operation.wait(10)
    await hass.async_block_till_done()
    assert fake.states["ollama"] == "up"
    assert events and events[0].data["kind"] == "launch"


async def test_status_service_returns_every_component(hass, powered) -> None:
    add_wyoming(hass)
    await powered()
    response = await hass.services.async_call(
        "yarnotron", "status", {}, blocking=True, return_response=True
    )
    sides = {c["side"] for c in response["components"]}
    assert sides == {"home_assistant", "mac"}
    assert response["ok"] is True
    assert "| Component | Where | Status |" in response["markdown"]


async def test_launch_while_one_is_running_joins_it(hass, supervisor, powered) -> None:
    fake, _ = supervisor
    entry = await powered()
    first = entry.runtime_data.power.async_launch()
    second = entry.runtime_data.power.async_launch()
    assert first is second
    await first.wait(10)
    assert fake.launch_calls == 1


# ------------------------------------------------------------ options flow


BASE_OPTIONS = {
    "scan_interval": 30,
    "idle_minutes": 30,
    "keep_listening": True,
    "include_transcripts": False,
    "voice_power": True,
}


async def _options(hass, entry, **extra):
    result = await hass.config_entries.options.async_init(entry.entry_id)
    return await hass.config_entries.options.async_configure(
        result["flow_id"], {**BASE_OPTIONS, **extra}
    )


async def test_options_save_a_working_supervisor(hass, supervisor, setup) -> None:
    _, url = supervisor
    entry = await setup()
    result = await _options(
        hass,
        entry,
        supervisor_url=url.removeprefix("http://"),
        supervisor_token=SUP_TOKEN,
        mac_address="AA-BB-CC-DD-EE-FF",
    )
    assert result["type"].value == "create_entry"
    await hass.async_block_till_done()
    assert entry.options["supervisor_url"] == url  # normalised
    assert entry.runtime_data.supervisor is not None
    assert entry.runtime_data.mac_address == "AA-BB-CC-DD-EE-FF"


async def test_options_reject_a_wrong_token_and_a_bad_mac(hass, supervisor, setup) -> None:
    _, url = supervisor
    entry = await setup()
    result = await _options(hass, entry, supervisor_url=url, supervisor_token="nope")
    assert result["errors"] == {"supervisor_token": "supervisor_auth"}
    result = await _options(
        hass, entry, supervisor_url=url, supervisor_token=SUP_TOKEN, mac_address="zz"
    )
    assert result["errors"] == {"mac_address": "invalid_mac"}
    result = await _options(hass, entry, mac_address="aa:bb:cc:dd:ee:ff")
    assert result["errors"] == {"supervisor_url": "supervisor_url_needed"}


async def test_options_accept_a_mac_that_is_asleep(hass, setup) -> None:
    entry = await setup()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = await _options(hass, entry, supervisor_url=f"127.0.0.1:{port}")
    assert result["type"].value == "create_entry"


async def test_clearing_the_supervisor_removes_mac_control(hass, supervisor, powered) -> None:
    entry = await powered()
    assert entry.runtime_data.supervisor_coordinator is not None
    result = await _options(hass, entry)
    assert result["type"].value == "create_entry"
    await hass.async_block_till_done()
    assert entry.runtime_data.supervisor is None


async def test_diagnostics_hide_the_supervisor_secrets(hass, supervisor, powered) -> None:
    from custom_components.yarnotron.diagnostics import async_get_config_entry_diagnostics

    _, url = supervisor
    entry = await powered(mac_address="aa:bb:cc:dd:ee:ff")
    dump = str(await async_get_config_entry_diagnostics(hass, entry))
    assert SUP_TOKEN not in dump and "aa:bb:cc:dd:ee:ff" not in dump and url not in dump
