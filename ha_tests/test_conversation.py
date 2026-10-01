from __future__ import annotations

from custom_components.yarnotron.const import (
    BACKEND_ERROR_TEXT,
    LOCKED_TEXT,
    STORYTIME_OFF_TEXT,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant

AGENT = "conversation.yarnotron"


async def _say(hass: HomeAssistant, text: str):
    result = await conversation.async_converse(
        hass, text, None, Context(), language="en", agent_id=AGENT
    )
    speech = result.response.speech["plain"]["speech"]
    return result, speech


async def test_story_turn_streams_the_backends_sentences(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    result, speech = await _say(hass, "tell me about a dragon")
    assert "Once upon a time." in speech and "A dragon yawned." in speech
    assert fake.turns[0]["transcript"] == "tell me about a dragon"
    assert fake.turns[0]["child_id"] == "local_child"
    assert fake.turns[0]["session_id"] == "sess_1"
    assert result.continue_conversation is True


async def test_second_turn_reuses_the_session(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    await _say(hass, "tell me about a dragon")
    await _say(hass, "make it blue")
    assert fake.sessions_created == 1
    assert fake.turns[1]["session_id"] == "sess_1"
    assert fake.turns[1]["story_id"] == "story_1"


async def test_reply_without_streamed_sentences_is_spoken(hass, backend, setup) -> None:
    fake, _ = backend
    fake.sentences = []
    fake.turn = {**fake.turn, "response_text": "Okay, undone."}
    await setup()
    _, speech = await _say(hass, "undo that")
    assert speech == "Okay, undone."


async def test_keep_listening_option_off(hass, backend, setup) -> None:
    await setup(keep_listening=False)
    result, _ = await _say(hass, "tell me a story")
    assert result.continue_conversation is False


async def test_storytime_off_never_reaches_backend(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    await hass.services.async_call(
        "switch", "turn_off", {"entity_id": "switch.yarnotron_storytime"}, blocking=True
    )
    result, speech = await _say(hass, "tell me a story")
    assert speech == STORYTIME_OFF_TEXT
    assert fake.turns == [] and fake.sessions_created == 0
    assert result.continue_conversation is False


async def test_controls_are_local(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    result, speech = await _say(hass, "help")
    assert "repeat that" in speech
    assert result.continue_conversation is True
    assert fake.turns == []


async def test_repeat_says_the_last_reply(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    await _say(hass, "tell me about a dragon")
    _, speech = await _say(hass, "repeat that")
    assert "A dragon yawned." in speech
    assert len(fake.turns) == 1


async def test_stop_ends_the_session_without_locking(hass, backend, setup) -> None:
    fake, _ = backend
    await setup()
    await _say(hass, "tell me about a dragon")
    result, _ = await _say(hass, "stop")
    assert fake.ended == ["sess_1"]
    assert result.continue_conversation is False
    assert hass.states.get("binary_sensor.yarnotron_session_locked").state == "off"
    await _say(hass, "another story")
    assert fake.sessions_created == 2


async def test_strike_limit_locks_until_a_parent_restarts(hass, backend, setup) -> None:
    fake, _ = backend
    fake.sentences = []
    fake.turn = {
        "response_text": "Let's take a break.",
        "safety": {"action": "block", "categories": ["strike_limit_reached"]},
    }
    await setup()
    result, _ = await _say(hass, "something awful")
    assert result.continue_conversation is False
    assert hass.states.get("binary_sensor.yarnotron_session_locked").state == "on"

    turns_before = len(fake.turns)
    _, speech = await _say(hass, "tell me a story")
    assert speech == LOCKED_TEXT
    assert len(fake.turns) == turns_before  # the lock holds: nothing reaches the backend

    fake.turn = {"response_text": "ok", "safety": {"action": "allow", "categories": []}}
    await hass.services.async_call("yarnotron", "start_new_session", {}, blocking=True)
    assert hass.states.get("binary_sensor.yarnotron_session_locked").state == "off"
    _, speech = await _say(hass, "tell me a story")
    assert speech != LOCKED_TEXT
    assert len(fake.turns) == turns_before + 1


async def test_lock_survives_a_restart(hass, backend, setup) -> None:
    fake, _ = backend
    fake.sentences = []
    fake.turn = {
        "response_text": "break",
        "safety": {"action": "block", "categories": ["session_ended"]},
    }
    entry = await setup()
    await _say(hass, "bad")
    await entry.runtime_data.sessions.async_save()

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    assert hass.states.get("binary_sensor.yarnotron_session_locked").state == "on"


async def test_idle_expiry_starts_a_new_session(hass, backend, setup) -> None:
    fake, _ = backend
    entry = await setup(idle_minutes=1)
    await _say(hass, "tell me about a dragon")
    sessions = entry.runtime_data.sessions
    sessions.state.last_activity = "2000-01-01T00:00:00+00:00"
    await _say(hass, "and then?")
    assert fake.ended == ["sess_1"]
    assert fake.sessions_created == 2


async def test_backend_down_gets_a_spoken_apology(hass, backend, setup) -> None:
    fake, _ = backend
    entry = await setup()
    fake.down = True
    result, speech = await _say(hass, "tell me a story")
    assert speech == BACKEND_ERROR_TEXT
    assert result.continue_conversation is False
    assert entry.runtime_data.sessions.locked is False


async def test_stream_error_is_apologised_not_raised(hass, backend, setup) -> None:
    fake, _ = backend
    fake.sentences = []
    fake.stream_error = {"error": "model unavailable", "status": 503}
    await setup()
    result, speech = await _say(hass, "tell me a story")
    assert BACKEND_ERROR_TEXT in speech
    assert result.continue_conversation is False


async def test_flagged_turn_alerts_after_followup(hass, backend, setup) -> None:
    fake, _ = backend
    fake.turn = {**fake.turn, "safety": {"action": "redirect", "categories": ["scary"]}}
    entry = await setup()
    await _say(hass, "tell me something scary")
    # The backend writes the event a moment after replying.
    fake.add_event("late", action="redirect", categories=["scary"])
    await entry.runtime_data.coordinator.async_refresh()
    await hass.async_block_till_done()
    assert hass.states.get("event.yarnotron_safety_alert").attributes["event_id"] == "late"


async def test_deleted_pinned_story_is_cleared(hass, backend, setup) -> None:
    fake, _ = backend
    entry = await setup()
    await hass.services.async_call(
        "yarnotron", "set_story", {"story_id": "story_gone"}, blocking=True
    )
    fake.create_session_404_for_story = True
    _, speech = await _say(hass, "tell me a story")
    assert speech != BACKEND_ERROR_TEXT
    assert entry.runtime_data.sessions.pinned_story_id is None
