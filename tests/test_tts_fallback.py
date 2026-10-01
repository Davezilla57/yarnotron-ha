from __future__ import annotations

from pathlib import Path

import pytest

from yarnotron.voice.tts import (
    FallbackSpeaker,
    NullSpeaker,
    kokoro_is_available,
    kokoro_model_paths,
)


class RecordingSpeaker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("engine unavailable")
        self.spoken.append(text)


def test_uses_the_first_working_engine():
    best, backup = RecordingSpeaker(), RecordingSpeaker()
    FallbackSpeaker([best, backup]).speak("Hello.")
    assert best.spoken == ["Hello."]
    assert backup.spoken == []


def test_falls_through_to_the_next_engine_on_failure():
    broken, backup = RecordingSpeaker(fail=True), RecordingSpeaker()
    FallbackSpeaker([broken, backup]).speak("Hello.")
    assert backup.spoken == ["Hello."]


def test_a_failed_engine_is_not_retried_every_sentence():
    """Retrying a dead engine would add its timeout to every single sentence."""

    class CountingFailure(RecordingSpeaker):
        def __init__(self) -> None:
            super().__init__(fail=True)
            self.attempts = 0

        def speak(self, text: str) -> None:
            self.attempts += 1
            raise RuntimeError("engine unavailable")

    broken, backup = CountingFailure(), RecordingSpeaker()
    speaker = FallbackSpeaker([broken, backup])
    for sentence in ("One.", "Two.", "Three."):
        speaker.speak(sentence)

    assert broken.attempts == 1
    assert backup.spoken == ["One.", "Two.", "Three."]


def test_total_failure_is_silent_rather_than_raising():
    """A dead synthesiser must not take down the turn loop."""
    speaker = FallbackSpeaker([RecordingSpeaker(fail=True), RecordingSpeaker(fail=True)])
    speaker.speak("Hello.")  # must not raise
    assert speaker.active is None


def test_active_reports_the_engine_in_use():
    broken, backup = RecordingSpeaker(fail=True), RecordingSpeaker()
    speaker = FallbackSpeaker([broken, backup])
    assert speaker.active is broken
    speaker.speak("Hello.")
    assert speaker.active is backup


def test_empty_speaker_list_is_rejected():
    with pytest.raises(ValueError):
        FallbackSpeaker([])


def test_kokoro_availability_is_a_file_check(tmp_path: Path):
    assert not kokoro_is_available(tmp_path)
    model, voices = kokoro_model_paths(tmp_path)
    model.write_bytes(b"x")
    assert not kokoro_is_available(tmp_path)  # needs both files
    voices.write_bytes(b"x")
    assert kokoro_is_available(tmp_path)


def test_null_speaker_still_satisfies_the_protocol():
    NullSpeaker().speak("anything")
