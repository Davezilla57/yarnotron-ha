from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import re
import time
from typing import Callable, Protocol

from yarnotron.core.timing import LatencyTimer, elapsed_ms
from yarnotron.voice.backend import BackendClient

DEFAULT_SPACE_FRIEND_HOWL_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "audio" / "space_friend_howl.m4a"
)


class UtteranceRecorder(Protocol):
    def record_utterance(self, listen_timeout_seconds: float | None = None) -> Path | None: ...


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class SoundEffectPlayer(Protocol):
    def play(self, path: Path) -> None: ...


class NullSoundEffectPlayer:
    def play(self, path: Path) -> None:
        return


@dataclass(frozen=True)
class VoiceLoopSettings:
    child_id: str = "local_child"
    story_id: str | None = None
    device_id: str = "macbook-local-voice"
    wake_word: str = "interpreter"
    wake_response: str = "I'm listening. What's your story?"
    idle_timeout_seconds: float = 10.0
    help_response: str = (
        "You can say things like: add a new character, change the castle to a treehouse, "
        "read me a recap, undo the last change, repeat that, pause, or stop listening."
    )
    sleep_response: str = "Okay, I'll listen for the wake word again."
    pause_response: str = "Paused. Say the wake word and resume when you're ready."
    resume_response: str = "I'm back. What's your story?"
    stop_response: str = "Okay, ending this Yarnotron session."
    repeat_unavailable_response: str = "I don't have anything to repeat yet."
    backend_error_response: str = (
        "I'm having trouble thinking right now. Please give me a moment, then try again."
    )
    space_friend_howl_path: Path | None = DEFAULT_SPACE_FRIEND_HOWL_PATH


@dataclass(frozen=True)
class VoiceTurn:
    transcript: str
    response_text: str


class VoiceControlIntent(str, Enum):
    HELP = "help"
    REPEAT = "repeat"
    SLEEP = "sleep"
    PAUSE = "pause"
    RESUME = "resume"
    STOP_SESSION = "stop_session"


@dataclass(frozen=True)
class VoiceControlResult:
    intent: VoiceControlIntent
    should_sleep: bool = False
    should_pause: bool = False
    should_resume: bool = False
    should_stop: bool = False


StatusCallback = Callable[[str], None]


class LocalVoiceLoop:
    def __init__(
        self,
        backend: BackendClient,
        recorder: UtteranceRecorder,
        transcriber: Transcriber,
        speaker: Speaker,
        sound_effect_player: SoundEffectPlayer | None = None,
        settings: VoiceLoopSettings | None = None,
        status: StatusCallback | None = None,
    ) -> None:
        self.backend = backend
        self.recorder = recorder
        self.transcriber = transcriber
        self.speaker = speaker
        self.sound_effect_player = sound_effect_player or NullSoundEffectPlayer()
        self.settings = settings or VoiceLoopSettings()
        self.status = status or (lambda _message: None)
        self.last_response_text: str | None = None
        self.interrupted = False

    def run_forever(self) -> None:
        settings = self.settings
        session = self.backend.create_session(
            settings.child_id,
            settings.story_id,
            settings.device_id,
        )
        self.status(f"Started session: {session['id']}")
        self.status(
            f"Listening for {settings.idle_timeout_seconds:g} seconds without a wake word. "
            f"After that, say '{settings.wake_word}' to wake Yarnotron. Use Ctrl-C to end."
        )

        try:
            awake = True
            paused = False
            should_stop = False
            while True:
                if should_stop:
                    break
                if not awake:
                    transcript = self.listen_for_transcript()
                    if not transcript:
                        continue
                    wake_payload = extract_wake_payload(transcript, settings.wake_word)
                    if wake_payload is None:
                        self.status("Wake word not heard.")
                        continue

                    awake = True
                    self.status(f"Wake word heard: {settings.wake_word}")
                    if not wake_payload:
                        if paused:
                            self.speaker.speak("Paused. Say resume when you're ready.")
                            awake = False
                        else:
                            self.speaker.speak(settings.wake_response)
                        continue

                    control = self.handle_control_intent(wake_payload, paused=paused)
                    if control is not None:
                        awake, paused = apply_control_state(
                            control,
                            awake=awake,
                            paused=paused,
                        )
                        should_stop = control.should_stop
                        continue
                    if paused:
                        self.speaker.speak("Paused. Say resume when you're ready.")
                        awake = False
                        continue

                    turn = self.run_turn(session_id=session["id"], transcript=wake_payload)
                else:
                    transcript = self.listen_for_transcript(
                        listen_timeout_seconds=settings.idle_timeout_seconds,
                    )
                    if not transcript:
                        turn = None
                    else:
                        control = self.handle_control_intent(transcript, paused=paused)
                        if control is not None:
                            awake, paused = apply_control_state(
                                control,
                                awake=awake,
                                paused=paused,
                            )
                            should_stop = control.should_stop
                            continue
                        turn = self.run_turn(session_id=session["id"], transcript=transcript)

                if turn is None:
                    self.status(
                        f"No speech for {settings.idle_timeout_seconds:g} seconds. "
                        f"Say '{settings.wake_word}' to wake me again."
                    )
                    awake = False
                    continue
                self.status(f"You: {turn.transcript}")
                self.status(f"Yarnotron: {turn.response_text}")
                self.last_response_text = turn.response_text
        finally:
            self.backend.end_session(session["id"])
            self.status(f"Ended session: {session['id']}")

    def listen_for_transcript(
        self,
        listen_timeout_seconds: float | None = None,
    ) -> str | None:
        timer = LatencyTimer(
            "voice_listen",
            listen_timeout_seconds=listen_timeout_seconds,
        )
        try:
            self.status("Listening...")
            with timer.measure("record"):
                audio_path = self.recorder.record_utterance(
                    listen_timeout_seconds=listen_timeout_seconds,
                )
            if audio_path is None:
                timer.add_metadata(result="no_speech")
                return None
            self.status("Transcribing...")
            try:
                with timer.measure("transcribe"):
                    transcript = self.transcriber.transcribe(audio_path).strip()
            finally:
                audio_path.unlink(missing_ok=True)
            timer.add_metadata(
                result="transcribed" if transcript else "empty_transcript",
                transcript_chars=len(transcript),
            )
            return transcript or None
        finally:
            timer.log()

    def run_turn(
        self,
        session_id: str,
        transcript: str | None = None,
        listen_timeout_seconds: float | None = None,
    ) -> VoiceTurn | None:
        settings = self.settings
        timer = LatencyTimer(
            "voice_turn",
            story_id=settings.story_id,
            session_id=session_id,
            device_id=settings.device_id,
        )
        try:
            if transcript is None:
                with timer.measure("listen_for_transcript"):
                    transcript = self.listen_for_transcript(
                        listen_timeout_seconds=listen_timeout_seconds,
                    )
            if not transcript:
                timer.add_metadata(result="no_transcript")
                return None

            timer.add_metadata(transcript_chars=len(transcript))
            control = self.handle_control_intent(transcript)
            if control is not None:
                timer.add_metadata(result="control", control_intent=control.intent.value)
                return None

            spoken: list[str] = []
            try:
                response = self._stream_backend_turn(
                    session_id=session_id,
                    transcript=transcript,
                    spoken=spoken,
                    timer=timer,
                )
            except RuntimeError as exc:
                timer.add_metadata(result="backend_error", error=str(exc))
                self.status(f"Backend turn failed: {exc}")
                with timer.measure("tts"):
                    self.speaker.speak(settings.backend_error_response)
                return None
            if response is None:
                timer.add_metadata(result="backend_error", error="stream ended without a turn")
                self.status("Backend turn failed: stream ended without a turn.")
                with timer.measure("tts"):
                    self.speaker.speak(settings.backend_error_response)
                return None
            response_text = str(response["response_text"])
            updated_story = response.get("updated_story")
            if isinstance(updated_story, dict) and isinstance(updated_story.get("id"), str):
                new_story_id = updated_story["id"]
                if new_story_id != settings.story_id:
                    self.settings = replace(settings, story_id=new_story_id)
                    settings = self.settings
                    timer.add_metadata(story_id=new_story_id)
                    self.status(f"Active story: {new_story_id}")
            timer.add_metadata(
                result="completed",
                intent=response.get("intent"),
                response_chars=len(response_text),
            )
            if spoken:
                # Already said aloud, sentence by sentence, while the model was
                # still writing. Nothing left to speak.
                timer.add_metadata(streamed_sentences=len(spoken))
            else:
                # Paths that never reach the model -- unknown, undo, media -- send
                # no sentences, so the reply is spoken here as it always was.
                with timer.measure("response_effect"):
                    self.play_response_effect(response)
                with timer.measure("tts"):
                    self.speaker.speak(response_text)
            return VoiceTurn(transcript=transcript, response_text=response_text)
        finally:
            timer.log()

    def _stream_backend_turn(
        self,
        session_id: str,
        transcript: str,
        spoken: list[str],
        timer: LatencyTimer,
    ) -> dict | None:
        """Speak sentences as they arrive; return the final turn payload."""
        settings = self.settings
        response: dict | None = None
        started = time.perf_counter()

        self.interrupted = False
        for event in self.backend.agent_turn_stream(
            child_id=settings.child_id,
            story_id=settings.story_id,
            session_id=session_id,
            device_id=settings.device_id,
            transcript=transcript,
        ):
            sentence = event.get("sentence")
            if isinstance(sentence, str) and sentence:
                if not spoken:
                    timer.add_metadata(first_audio_ms=elapsed_ms(started))
                spoken.append(sentence)
                if self.interrupted:
                    continue  # already handed the floor over
                if self._speak(sentence):
                    # The child talked over us. Stop the reply here and let them
                    # have the floor; the rest of the stream is drained without
                    # being spoken so the turn still records what was generated.
                    self.interrupted = True
                    timer.add_metadata(barged_in_after_sentences=len(spoken))
                    self.status("Interrupted, listening.")
                continue
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            turn = event.get("turn")
            if isinstance(turn, dict):
                response = turn
        return response

    def _speak(self, text: str) -> bool:
        """Speak one utterance. True if the child interrupted it."""
        interruptible = getattr(self.speaker, "speak_until_interrupted", None)
        if interruptible is not None:
            return bool(interruptible(text))
        self.speaker.speak(text)
        return False

    def play_response_effect(self, response: dict) -> None:
        if response.get("intent") != "unknown":
            return
        path = self.settings.space_friend_howl_path
        if path is None:
            return
        self.status("Playing space-friend howl.")
        self.sound_effect_player.play(path)

    def handle_control_intent(
        self,
        transcript: str,
        *,
        paused: bool = False,
    ) -> VoiceControlResult | None:
        intent = classify_control_intent(transcript, paused=paused)
        if intent is None:
            return None

        timer = LatencyTimer("voice_control", intent=intent.value)
        settings = self.settings
        try:
            match intent:
                case VoiceControlIntent.HELP:
                    response_text = settings.help_response
                    result = VoiceControlResult(intent=intent)
                case VoiceControlIntent.REPEAT:
                    response_text = self.last_response_text or settings.repeat_unavailable_response
                    result = VoiceControlResult(intent=intent)
                case VoiceControlIntent.SLEEP:
                    response_text = settings.sleep_response
                    result = VoiceControlResult(intent=intent, should_sleep=True)
                case VoiceControlIntent.PAUSE:
                    response_text = settings.pause_response
                    result = VoiceControlResult(intent=intent, should_pause=True)
                case VoiceControlIntent.RESUME:
                    response_text = settings.resume_response
                    result = VoiceControlResult(intent=intent, should_resume=True)
                case VoiceControlIntent.STOP_SESSION:
                    response_text = settings.stop_response
                    result = VoiceControlResult(intent=intent, should_stop=True)

            self.status(f"Control intent: {intent.value}")
            timer.add_metadata(response_chars=len(response_text))
            with timer.measure("tts"):
                self.speaker.speak(response_text)
            return result
        finally:
            timer.log()


def extract_wake_payload(transcript: str, wake_word: str) -> str | None:
    if not wake_word.strip():
        return transcript.strip()

    pattern = re.compile(rf"\b{re.escape(wake_word.strip())}\b", re.IGNORECASE)
    match = pattern.search(transcript)
    if match is None:
        return None

    payload = transcript[match.end() :].strip(" ,.!?:;-")
    return payload.strip()


def classify_control_intent(
    transcript: str,
    *,
    paused: bool = False,
) -> VoiceControlIntent | None:
    normalized = normalize_control_text(transcript)
    if not normalized:
        return None

    if normalized in {
        "help",
        "help me",
        "what can i say",
        "what can you do",
        "how does this work",
        "tell me what i can say",
    }:
        return VoiceControlIntent.HELP

    if normalized in {
        "repeat",
        "repeat that",
        "say that again",
        "say it again",
        "what did you say",
    }:
        return VoiceControlIntent.REPEAT

    if normalized in {
        "sleep",
        "go to sleep",
        "stop listening",
        "that's all",
        "that is all",
        "never mind",
    }:
        return VoiceControlIntent.SLEEP

    if normalized in {"pause", "pause yarnotron", "hold on", "wait a second"}:
        return VoiceControlIntent.PAUSE

    if paused and normalized in {
        "resume",
        "continue",
        "keep going",
        "i'm ready",
        "im ready",
        "ready",
    }:
        return VoiceControlIntent.RESUME

    if normalized in {
        "stop",
        "stop session",
        "end session",
        "quit",
        "exit",
        "goodbye",
    }:
        return VoiceControlIntent.STOP_SESSION

    return None


def normalize_control_text(transcript: str) -> str:
    text = transcript.casefold().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_control_state(
    control: VoiceControlResult,
    *,
    awake: bool,
    paused: bool,
) -> tuple[bool, bool]:
    if control.should_resume:
        return True, False
    if control.should_pause:
        return False, True
    if control.should_sleep:
        return False, False
    return awake, paused
