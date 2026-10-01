"""Barge-in detector, on synthetic signals so it runs without audio hardware.

The constants were tuned against a real recording of Yarnotron speaking through
MacBook speakers -- echo peaking around 0.10 rms while a child near the
microphone sits around 0.11 median. These tests reproduce that shape rather
than that recording, so they are deterministic and need no fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from yarnotron.voice.bargein import BargeInDetector, BargeInSettings

SR = 16_000
FRAME = 512
rng = np.random.default_rng(0)


def noise(level: float, frames: int = 1) -> np.ndarray:
    return (rng.standard_normal(FRAME * frames) * level).astype(np.float32)


def feed(detector: BargeInDetector, signal: np.ndarray) -> float | None:
    """Returns the time at which it fired, or None."""
    for index in range(0, (len(signal) // FRAME) * FRAME, FRAME):
        if detector.feed(signal[index : index + FRAME]):
            return (index + FRAME) / SR
    return None


def test_our_own_reply_does_not_interrupt_us():
    """The whole difficulty: the microphone hears Yarnotron at 30x room level."""
    detector = BargeInDetector()
    echo = noise(0.05, frames=90)  # steady echo, well above a quiet room
    assert feed(detector, echo) is None


def test_a_child_talking_over_us_interrupts():
    detector = BargeInDetector()
    signal = np.concatenate([noise(0.05, frames=30), noise(0.25, frames=40)])
    fired = feed(detector, signal)
    assert fired is not None
    # Shortly after the louder speech starts, not instantly and not never.
    assert 0.95 < fired < 1.6


def test_nothing_can_interrupt_during_the_grace_window():
    """The window measures the echo, and covers the tail of the last utterance."""
    settings = BargeInSettings(grace_seconds=0.4)
    detector = BargeInDetector(settings)
    assert feed(detector, noise(0.9, frames=10)) is None


def test_threshold_follows_the_echo_rather_than_being_fixed():
    quiet = BargeInDetector()
    feed(quiet, noise(0.02, frames=13))
    loud = BargeInDetector()
    feed(loud, noise(0.20, frames=13))
    assert loud.threshold > quiet.threshold * 3


def test_a_brief_noise_is_not_an_interruption():
    """A cough or a chair scrape must not cancel the story."""
    detector = BargeInDetector()
    signal = np.concatenate([
        noise(0.05, frames=20),
        noise(0.4, frames=2),      # short bang
        noise(0.05, frames=30),
    ])
    assert feed(detector, signal) is None


def test_reset_clears_state_between_utterances():
    detector = BargeInDetector()
    assert feed(detector, np.concatenate([noise(0.05, 30), noise(0.25, 40)])) is not None
    assert detector.triggered
    detector.reset()
    assert not detector.triggered
    assert feed(detector, noise(0.05, frames=60)) is None


def test_stays_triggered_once_it_fires():
    detector = BargeInDetector()
    feed(detector, np.concatenate([noise(0.05, 30), noise(0.25, 40)]))
    assert detector.feed(noise(0.001)) is True


def test_a_vad_that_rejects_speech_suppresses_the_trigger():
    """Loud but not speech-shaped -- a vacuum cleaner, not a child."""
    detector = BargeInDetector(vad=lambda frame: np.array([[0.0]]))
    signal = np.concatenate([noise(0.05, frames=30), noise(0.25, frames=40)])
    assert feed(detector, signal) is None


def test_a_broken_vad_does_not_disable_barge_in():
    def exploding(frame):
        raise RuntimeError("model gone")

    detector = BargeInDetector(vad=exploding)
    signal = np.concatenate([noise(0.05, frames=30), noise(0.25, frames=40)])
    assert feed(detector, signal) is not None


@pytest.mark.parametrize("level", [0.05, 0.12, 0.30])
def test_calibration_scales_with_playback_volume(level):
    """Turning the speakers up must not make the child unable to interrupt."""
    detector = BargeInDetector()
    signal = np.concatenate([noise(level, frames=30), noise(level * 5, frames=40)])
    assert feed(detector, signal) is not None


def test_room_noise_just_above_the_echo_is_not_an_interruption():
    """The regression: 4 of 22 replies were cut short in a real session.

    At the old margin of 1.1, anything a tenth louder than our own echo counted,
    which ordinary room noise clears. A voice has to be clearly nearer now.
    """
    detector = BargeInDetector()
    signal = np.concatenate([noise(0.05, frames=30), noise(0.065, frames=40)])

    assert feed(detector, signal) is None


def test_a_child_still_gets_through_at_the_new_margin() -> None:
    """Quieter than the test above but still clearly nearer than the speakers."""
    detector = BargeInDetector()
    signal = np.concatenate([noise(0.05, frames=30), noise(0.12, frames=40)])

    assert feed(detector, signal) is not None


def test_the_margin_can_be_tuned_per_room() -> None:
    signal = np.concatenate([noise(0.05, frames=30), noise(0.065, frames=40)])

    # The same noise that the default ignores interrupts an eager detector.
    assert feed(BargeInDetector(BargeInSettings(margin=1.1)), signal) is not None
    assert feed(BargeInDetector(BargeInSettings(margin=1.4)), signal) is None


def test_requiring_more_sustained_speech_can_be_tuned_too() -> None:
    # Six frames of speech: enough at the old requirement, not at the new one.
    signal = np.concatenate([
        noise(0.05, frames=20),
        noise(0.25, frames=6),
        noise(0.05, frames=30),
    ])

    assert feed(BargeInDetector(BargeInSettings(speech_frames_required=6)), signal) is not None
    assert feed(BargeInDetector(BargeInSettings(speech_frames_required=8)), signal) is None
