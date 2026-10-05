"""The Assist conversation agent: Home Assistant's ears and voice, Yarnotron's brain.

A Home Assistant voice satellite does wake word, speech-to-text and
text-to-speech. This agent sits between the pipeline's text in and text out and
talks to the Yarnotron backend, which keeps everything that makes it Yarnotron:
the story memory, the safety review and the parent controls.

Replies are streamed. The backend releases each safety-gated sentence as it is
written; this agent feeds them into the chat log as they arrive, so a streaming
TTS engine can start speaking before the reply is finished. An engine that
cannot stream simply speaks the whole reply at the end.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from homeassistant.components import conversation, persistent_notification
from homeassistant.components.conversation import AssistantContentDeltaDict
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .api import YarnotronApiError, YarnotronAuthError, YarnotronError
from .const import (
    BACKEND_ERROR_TEXT,
    CONF_KEEP_LISTENING,
    CONF_VOICE_POWER,
    DEFAULT_KEEP_LISTENING,
    DEFAULT_VOICE_POWER,
    EMPTY_REPLY_TEXT,
    HELP_TEXT,
    LOCKED_TEXT,
    NOTHING_TO_REPEAT_TEXT,
    POWER_DISABLED_TEXT,
    POWERED_DOWN_TEXT,
    SHUTDOWN_CANCELLED_TEXT,
    SHUTDOWN_CONFIRM_SECONDS,
    SHUTDOWN_CONFIRM_TEXT,
    SLEEP_TEXT,
    STOP_TEXT,
    STORYTIME_OFF_TEXT,
    VOICE_POWER_WAIT,
)
from .controls import POWER_INTENTS, ControlIntent, classify, is_confirmation
from .entity import device_info_for
from .models import YarnotronConfigEntry, YarnotronRuntimeData
from .session import SessionLockedError

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: YarnotronConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([YarnotronConversationEntity(entry)])


@dataclass
class _TurnOutcome:
    """What happened while a turn streamed, for the code after the stream."""

    turn: dict[str, Any] | None = None
    error: YarnotronError | None = None
    streamed: int = 0
    spoken: list[str] = field(default_factory=list)


class YarnotronConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Yarnotron as an Assist conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None  # takes the device's name
    _attr_supports_streaming = True

    def __init__(self, entry: YarnotronConfigEntry) -> None:
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = device_info_for(entry)

    # ------------------------------------------------------------ registration

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        # The backend's prompts, safety review and the original voice stack are
        # all English. Declaring more would promise what it cannot do.
        return ["en"]

    @property
    def _runtime(self) -> YarnotronRuntimeData:
        return self.entry.runtime_data

    # ---------------------------------------------------------------- handling

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        runtime = self._runtime

        # A spoken "shut down" asks for a yes first. Whatever comes next either
        # confirms it, declines it, or is something else and cancels it quietly.
        if runtime.shutdown_confirm_until:
            pending = time.monotonic() < runtime.shutdown_confirm_until
            runtime.shutdown_confirm_until = 0.0
            answer = is_confirmation(user_input.text) if pending else None
            if answer is True:
                return await self._power_shutdown(user_input, chat_log)
            if answer is False:
                return self._say(user_input, chat_log, SHUTDOWN_CANCELLED_TEXT, keep_listening=False)

        control = classify(user_input.text)

        # Starting, stopping and checking the machines is not storytime, so these
        # work while Storytime is off and while a session is locked.
        if control in POWER_INTENTS:
            return await self._handle_power(user_input, chat_log, control)

        if not runtime.storytime_enabled:
            return self._say(user_input, chat_log, STORYTIME_OFF_TEXT, keep_listening=False)

        if runtime.powered_down:
            return self._say(user_input, chat_log, POWERED_DOWN_TEXT, keep_listening=False)

        if control is not None:
            return await self._handle_control(user_input, chat_log, control)

        return await self._handle_story_turn(user_input, chat_log)

    # ------------------------------------------------------------- power

    async def _handle_power(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        control: ControlIntent,
    ) -> conversation.ConversationResult:
        runtime = self._runtime
        power = runtime.power
        assert power is not None

        if control is ControlIntent.STATUS:
            report = await power.async_status()
            if not report.ok or any(c.issue for c in report.components):
                self._notify_status(report)
            return self._say(user_input, chat_log, report.spoken(), keep_listening=False)

        if not bool(self.entry.options.get(CONF_VOICE_POWER, DEFAULT_VOICE_POWER)):
            return self._say(user_input, chat_log, POWER_DISABLED_TEXT, keep_listening=False)

        if control is ControlIntent.SHUTDOWN:
            runtime.shutdown_confirm_until = time.monotonic() + SHUTDOWN_CONFIRM_SECONDS
            return self._say(user_input, chat_log, SHUTDOWN_CONFIRM_TEXT, keep_listening=True)

        operation = power.async_launch()
        return await self._await_power(user_input, chat_log, operation, "Starting up now.")

    async def _power_shutdown(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        power = self._runtime.power
        assert power is not None
        operation = power.async_shutdown()
        return await self._await_power(user_input, chat_log, operation, "Shutting down now.")

    async def _await_power(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        operation,
        working_text: str,
    ) -> conversation.ConversationResult:
        """Wait a while for the result; if it is slow, say so and let it finish by itself."""
        result = await operation.wait(VOICE_POWER_WAIT)
        if result is None:
            text = (
                f"{working_text} This is taking a while. I'll keep going, and you can ask "
                "for the Yarnotron status in a minute."
            )
        else:
            text = result.spoken
        return self._say(user_input, chat_log, text, keep_listening=False)

    def _notify_status(self, report) -> None:
        persistent_notification.async_create(
            self.hass,
            report.markdown(),
            title="Yarnotron status: something needs attention",
            notification_id=f"{self.entry.domain}_status_{self.entry.entry_id}",
        )

    async def _handle_control(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        control: ControlIntent,
    ) -> conversation.ConversationResult:
        runtime = self._runtime
        match control:
            case ControlIntent.HELP:
                return self._say(user_input, chat_log, HELP_TEXT, keep_listening=True)
            case ControlIntent.REPEAT:
                return self._say(
                    user_input,
                    chat_log,
                    runtime.last_response or NOTHING_TO_REPEAT_TEXT,
                    keep_listening=True,
                )
            case ControlIntent.SLEEP:
                return self._say(user_input, chat_log, SLEEP_TEXT, keep_listening=False)
            case ControlIntent.STOP_SESSION:
                await runtime.sessions.async_end_session()
                return self._say(user_input, chat_log, STOP_TEXT, keep_listening=False)
        raise AssertionError(f"unhandled control {control}")  # pragma: no cover

    async def _handle_story_turn(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        runtime = self._runtime
        device_label = self._device_label(user_input)

        try:
            session_id, story_id = await runtime.sessions.async_prepare_turn(device_label)
        except SessionLockedError:
            # No turn reaches the backend until a grown-up starts a new session.
            return self._say(user_input, chat_log, LOCKED_TEXT, keep_listening=False)
        except YarnotronAuthError:
            self.entry.async_start_reauth(self.hass)
            return self._say(user_input, chat_log, BACKEND_ERROR_TEXT, keep_listening=False)
        except YarnotronError as err:
            _LOGGER.warning("Could not start a Yarnotron session: %s", err)
            return self._say(user_input, chat_log, BACKEND_ERROR_TEXT, keep_listening=False)

        outcome = _TurnOutcome()
        stream = self._deltas(user_input.text, session_id, story_id, device_label, outcome)
        async for _content in chat_log.async_add_delta_content_stream(self.entity_id, stream):
            pass

        result = conversation.async_get_result_from_chat_log(user_input, chat_log)

        if outcome.error is not None:
            if isinstance(outcome.error, YarnotronAuthError):
                self.entry.async_start_reauth(self.hass)
            result.continue_conversation = False
            return result

        turn = outcome.turn
        assert turn is not None  # _deltas records an error when there is no turn
        await runtime.sessions.async_record_turn(turn)
        runtime.last_response = " ".join(outcome.spoken).strip() or None

        coordinator = runtime.coordinator
        self.hass.async_create_task(coordinator.async_request_refresh())
        if (turn.get("safety") or {}).get("action", "allow") != "allow":
            coordinator.async_schedule_followup()

        result.continue_conversation = self._keep_listening and not runtime.sessions.locked
        return result

    async def _deltas(
        self,
        transcript: str,
        session_id: str,
        story_id: str | None,
        device_label: str,
        outcome: _TurnOutcome,
    ) -> AsyncGenerator[AssistantContentDeltaDict]:
        """Turn the backend's NDJSON events into chat-log deltas.

        Errors become a spoken apology inside the stream rather than an
        exception: once text has started reaching the speaker, a stream that
        raises would leave the pipeline with half a sentence and no ending.
        """
        yield {"role": "assistant"}
        runtime = self._runtime
        first = True

        def piece(text: str) -> str:
            nonlocal first
            out = text if first else f" {text}"
            first = False
            return out

        try:
            async for event in runtime.api.stream_turn(
                child_id=runtime.coordinator.child_id,
                story_id=story_id,
                session_id=session_id,
                device_id=device_label,
                transcript=transcript,
            ):
                sentence = event.get("sentence")
                if isinstance(sentence, str) and sentence.strip():
                    outcome.streamed += 1
                    outcome.spoken.append(sentence.strip())
                    yield {"content": piece(sentence.strip())}
                    continue
                if "error" in event:
                    raise YarnotronApiError(int(event.get("status") or 500), str(event["error"]))
                turn = event.get("turn")
                if isinstance(turn, dict):
                    outcome.turn = turn

            if outcome.turn is None:
                raise YarnotronApiError(502, "the stream ended without a turn")
        except YarnotronError as err:
            _LOGGER.warning("Yarnotron turn failed: %s", err)
            outcome.error = err
            yield {"content": piece(BACKEND_ERROR_TEXT)}
            return

        if outcome.streamed:
            # Already spoken, sentence by sentence, while the model was writing.
            return

        # Paths that never reach the model -- an unknown request, an undo, an
        # image request, a safety stop -- stream nothing, so say the reply here.
        reply = str(outcome.turn.get("response_text") or "").strip() or EMPTY_REPLY_TEXT
        outcome.spoken.append(reply)
        yield {"content": reply}

    # ----------------------------------------------------------------- helpers

    @property
    def _keep_listening(self) -> bool:
        return bool(self.entry.options.get(CONF_KEEP_LISTENING, DEFAULT_KEEP_LISTENING))

    def _say(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        text: str,
        *,
        keep_listening: bool,
    ) -> conversation.ConversationResult:
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=self.entity_id, content=text)
        )
        result = conversation.async_get_result_from_chat_log(user_input, chat_log)
        # The chat log would continue only after a question. A story session is
        # a back-and-forth, so keep the satellite listening unless this reply
        # ends the exchange (stop, sleep, locked, error).
        result.continue_conversation = keep_listening and self._keep_listening
        return result

    def _device_label(self, user_input: conversation.ConversationInput) -> str:
        """A name a parent will recognise on the dashboard's session list."""
        if user_input.device_id:
            device = dr.async_get(self.hass).async_get(user_input.device_id)
            if device is not None:
                return device.name_by_user or device.name or user_input.device_id
        return user_input.satellite_id or "home-assistant"
