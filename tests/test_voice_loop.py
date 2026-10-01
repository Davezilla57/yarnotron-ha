from __future__ import annotations

from pathlib import Path

from yarnotron.voice.loop import (
    LocalVoiceLoop,
    VoiceControlIntent,
    VoiceControlResult,
    VoiceLoopSettings,
    apply_control_state,
    classify_control_intent,
    extract_wake_payload,
)


class FakeRecorder:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path

    def record_utterance(self, listen_timeout_seconds: float | None = None) -> Path | None:
        self.audio_path.write_bytes(b"fake wav")
        return self.audio_path


class SilentRecorder:
    def record_utterance(self, listen_timeout_seconds: float | None = None) -> Path | None:
        assert listen_timeout_seconds == 10.0
        return None


class OneShotStopRecorder:
    """Fails if listened to more than once, so a wake-word regression hangs no test."""

    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.calls = 0

    def record_utterance(self, listen_timeout_seconds: float | None = None) -> Path | None:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("listened more than once; wake-word gate was not bypassed on startup")
        assert listen_timeout_seconds == 10.0
        self.audio_path.write_bytes(b"fake wav")
        return self.audio_path


class StopTranscriber:
    def transcribe(self, audio_path: Path) -> str:
        return "stop"


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> str:
        assert audio_path.exists()
        return "Add a moonlit tower to the story."


class FakeBackend:
    def __init__(self, response_payload: dict[str, object] | None = None) -> None:
        self.turn_payloads: list[dict[str, str | None]] = []
        self.response_payload = response_payload or {
            "response_text": "The tower catches silver moonlight."
        }
        self.stream_sentences: list[str] = []

    def create_session(
        self,
        child_id: str,
        story_id: str | None,
        device_id: str,
    ) -> dict[str, str]:
        return {"id": "session_test"}

    def end_session(self, session_id: str) -> dict[str, str]:
        return {"id": session_id}

    def agent_turn(
        self,
        child_id: str,
        story_id: str | None,
        session_id: str,
        device_id: str,
        transcript: str,
    ) -> dict[str, object]:
        self.turn_payloads.append(
            {
                "child_id": child_id,
                "story_id": story_id,
                "session_id": session_id,
                "device_id": device_id,
                "transcript": transcript,
            }
        )
        return self.response_payload

    def agent_turn_stream(
        self,
        child_id: str,
        story_id: str | None,
        session_id: str,
        device_id: str,
        transcript: str,
    ):
        """Default fake takes the no-sentence path, as the non-model intents do."""
        payload = self.agent_turn(child_id, story_id, session_id, device_id, transcript)
        for sentence in self.stream_sentences:
            yield {"sentence": sentence}
        yield {"turn": payload}


class FailingBackend(FakeBackend):
    def agent_turn(
        self,
        child_id: str,
        story_id: str | None,
        session_id: str,
        device_id: str,
        transcript: str,
    ) -> dict[str, object]:
        raise RuntimeError("Ollama is still warming up")

    def agent_turn_stream(self, *args: object, **kwargs: object):
        raise RuntimeError("Ollama is still warming up")
        yield  # pragma: no cover - generator marker


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeSoundEffectPlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []

    def play(self, path: Path) -> None:
        self.played.append(path)


def test_voice_loop_runs_turn_against_existing_backend_contract(tmp_path: Path) -> None:
    backend = FakeBackend()
    speaker = FakeSpeaker()
    audio_path = tmp_path / "utterance.wav"
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=FakeRecorder(audio_path),
        transcriber=FakeTranscriber(),
        speaker=speaker,
        settings=VoiceLoopSettings(
            child_id="child_test",
            story_id="story_test",
            device_id="device_test",
        ),
    )

    turn = loop.run_turn(session_id="session_test")

    assert turn is not None
    assert turn.transcript == "Add a moonlit tower to the story."
    assert turn.response_text == "The tower catches silver moonlight."
    assert backend.turn_payloads == [
        {
            "child_id": "child_test",
            "story_id": "story_test",
            "session_id": "session_test",
            "device_id": "device_test",
            "transcript": "Add a moonlit tower to the story.",
        }
    ]
    assert speaker.spoken == ["The tower catches silver moonlight."]
    assert not audio_path.exists()


def test_voice_loop_keeps_backend_created_story_id_for_later_turns() -> None:
    backend = FakeBackend(
        {
            "intent": "continue_story",
            "response_text": "I started The Moon Clock.",
            "updated_story": {"id": "story_the_moon_clock", "title": "The Moon Clock"},
        }
    )
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
        settings=VoiceLoopSettings(story_id=None),
    )

    first_turn = loop.run_turn(session_id="session_test", transcript="Start The Moon Clock.")
    second_turn = loop.run_turn(session_id="session_test", transcript="Add a glowing key.")

    assert first_turn is not None
    assert second_turn is not None
    assert [payload["story_id"] for payload in backend.turn_payloads] == [
        None,
        "story_the_moon_clock",
    ]


def test_voice_loop_plays_space_friend_howl_for_unknown_intent(tmp_path: Path) -> None:
    backend = FakeBackend(
        {
            "intent": "unknown",
            "response_text": "I did not quite catch how that changes the story.",
        }
    )
    speaker = FakeSpeaker()
    sound_effects = FakeSoundEffectPlayer()
    howl_path = tmp_path / "space_friend_howl.m4a"
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
        sound_effect_player=sound_effects,
        settings=VoiceLoopSettings(space_friend_howl_path=howl_path),
    )

    turn = loop.run_turn(session_id="session_test", transcript="huh")

    assert turn is not None
    assert sound_effects.played == [howl_path]
    assert speaker.spoken == ["I did not quite catch how that changes the story."]


def test_voice_loop_survives_backend_turn_error() -> None:
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=FailingBackend(),
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
    )

    turn = loop.run_turn(session_id="session_test", transcript="Add a moonlit tower.")

    assert turn is None
    assert speaker.spoken == [loop.settings.backend_error_response]


def test_voice_loop_returns_none_after_listening_timeout() -> None:
    backend = FakeBackend()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=FakeSpeaker(),
        settings=VoiceLoopSettings(idle_timeout_seconds=10.0),
    )

    turn = loop.run_turn(session_id="session_test", listen_timeout_seconds=10.0)

    assert turn is None
    assert backend.turn_payloads == []


def test_voice_loop_listens_without_a_wake_word_on_startup(tmp_path: Path) -> None:
    backend = FakeBackend()
    recorder = OneShotStopRecorder(tmp_path / "utterance.wav")
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=recorder,
        transcriber=StopTranscriber(),
        speaker=FakeSpeaker(),
        settings=VoiceLoopSettings(idle_timeout_seconds=10.0),
    )

    loop.run_forever()

    assert recorder.calls == 1
    assert backend.turn_payloads == []


def test_extract_wake_payload_requires_wake_word() -> None:
    assert extract_wake_payload("hello yarnotron", "interpreter") is None
    assert extract_wake_payload("interpreter", "interpreter") == ""
    assert (
        extract_wake_payload("Hey interpreter, add a moonlit tower.", "interpreter")
        == "add a moonlit tower"
    )


def test_classify_control_intent_handles_common_voice_commands() -> None:
    assert classify_control_intent("What can I say?") == VoiceControlIntent.HELP
    assert classify_control_intent("say that again") == VoiceControlIntent.REPEAT
    assert classify_control_intent("stop listening") == VoiceControlIntent.SLEEP
    assert classify_control_intent("hold on") == VoiceControlIntent.PAUSE
    assert classify_control_intent("resume") is None
    assert classify_control_intent("resume", paused=True) == VoiceControlIntent.RESUME
    assert classify_control_intent("end session") == VoiceControlIntent.STOP_SESSION
    assert classify_control_intent("add a dragon with a silver hat") is None


def test_voice_control_help_does_not_call_backend() -> None:
    backend = FakeBackend()
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
    )

    turn = loop.run_turn(session_id="session_test", transcript="help")

    assert turn is None
    assert backend.turn_payloads == []
    assert speaker.spoken == [loop.settings.help_response]


def test_voice_control_repeat_uses_last_response() -> None:
    backend = FakeBackend()
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
    )
    loop.last_response_text = "The moonlit tower sparkles."

    result = loop.handle_control_intent("repeat that")

    assert result is not None
    assert result.intent == VoiceControlIntent.REPEAT
    assert backend.turn_payloads == []
    assert speaker.spoken == ["The moonlit tower sparkles."]


def test_apply_control_state_separates_sleep_pause_and_resume() -> None:
    assert apply_control_state(
        VoiceControlResult(VoiceControlIntent.SLEEP, should_sleep=True),
        awake=True,
        paused=True,
    ) == (False, False)
    assert apply_control_state(
        VoiceControlResult(VoiceControlIntent.PAUSE, should_pause=True),
        awake=True,
        paused=False,
    ) == (False, True)
    assert apply_control_state(
        VoiceControlResult(VoiceControlIntent.RESUME, should_resume=True),
        awake=False,
        paused=True,
    ) == (True, False)


def test_voice_loop_speaks_each_streamed_sentence_as_it_arrives() -> None:
    backend = FakeBackend(
        {
            "intent": "continue_story",
            "response_text": "Oh, a seal pup! What does Wren do next?",
        }
    )
    backend.stream_sentences = ["Oh, a seal pup!", "What does Wren do next?"]
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
        settings=VoiceLoopSettings(story_id="story_test"),
    )

    turn = loop.run_turn(session_id="session_test", transcript="It is a seal pup.")

    assert turn is not None
    # Spoken as two utterances while the model was still writing, and the whole
    # reply is not repeated afterwards.
    assert speaker.spoken == ["Oh, a seal pup!", "What does Wren do next?"]


def test_voice_loop_speaks_once_when_no_sentences_are_streamed() -> None:
    """Non-model intents send no sentences, so the reply is spoken as before."""
    backend = FakeBackend({"intent": "undo_last_change", "response_text": "I undid that."})
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
    )

    loop.run_turn(session_id="session_test", transcript="undo that")

    assert speaker.spoken == ["I undid that."]


def test_voice_loop_reports_an_error_event_from_the_stream() -> None:
    class ErroringBackend(FakeBackend):
        def agent_turn_stream(self, *args: object, **kwargs: object):
            yield {"sentence": "Starting..."}
            yield {"error": "ollama down", "status": 503}

    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=ErroringBackend(),
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
    )

    turn = loop.run_turn(session_id="session_test", transcript="Add a tower.")

    assert turn is None
    assert speaker.spoken[-1] == loop.settings.backend_error_response


class InterruptingSpeaker(FakeSpeaker):
    """Reports an interruption on the Nth utterance, like a child talking over."""

    def __init__(self, interrupt_on: int) -> None:
        super().__init__()
        self.interrupt_on = interrupt_on

    def speak_until_interrupted(self, text: str) -> bool:
        self.spoken.append(text)
        return len(self.spoken) >= self.interrupt_on


def test_barge_in_stops_the_reply_where_the_child_cut_in() -> None:
    backend = FakeBackend(
        {"intent": "continue_story", "response_text": "One. Two. Three."}
    )
    backend.stream_sentences = ["One.", "Two.", "Three."]
    speaker = InterruptingSpeaker(interrupt_on=2)
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
        settings=VoiceLoopSettings(story_id="story_test"),
    )

    turn = loop.run_turn(session_id="session_test", transcript="Tell me a story.")

    assert turn is not None
    # Spoke up to the interruption and then stopped; the third never plays.
    assert speaker.spoken == ["One.", "Two."]
    assert loop.interrupted is True


def test_the_turn_still_completes_after_an_interruption() -> None:
    """The reply was cut short, but the turn is still recorded and returned."""
    backend = FakeBackend(
        {"intent": "continue_story", "response_text": "One. Two. Three."}
    )
    backend.stream_sentences = ["One.", "Two.", "Three."]
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=InterruptingSpeaker(interrupt_on=1),
        settings=VoiceLoopSettings(story_id="story_test"),
    )

    turn = loop.run_turn(session_id="session_test", transcript="Tell me a story.")

    assert turn is not None
    assert turn.response_text == "One. Two. Three."


def test_interrupted_flag_resets_between_turns() -> None:
    backend = FakeBackend({"intent": "continue_story", "response_text": "One."})
    backend.stream_sentences = ["One."]
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=InterruptingSpeaker(interrupt_on=1),
        settings=VoiceLoopSettings(story_id="story_test"),
    )
    loop.run_turn(session_id="session_test", transcript="First.")
    assert loop.interrupted is True

    loop.speaker = FakeSpeaker()  # no interruption this time
    loop.run_turn(session_id="session_test", transcript="Second.")
    assert loop.interrupted is False


def test_a_speaker_without_barge_in_support_still_works() -> None:
    """MacSaySpeaker and friends have no speak_until_interrupted."""
    backend = FakeBackend({"intent": "continue_story", "response_text": "One. Two."})
    backend.stream_sentences = ["One.", "Two."]
    speaker = FakeSpeaker()
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=SilentRecorder(),
        transcriber=FakeTranscriber(),
        speaker=speaker,
        settings=VoiceLoopSettings(story_id="story_test"),
    )

    loop.run_turn(session_id="session_test", transcript="Tell me a story.")

    assert speaker.spoken == ["One.", "Two."]
    assert loop.interrupted is False
