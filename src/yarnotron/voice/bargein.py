"""Decide whether the child has started talking over the reply.

Barge-in means listening while speaking, and on speakers the microphone hears
Yarnotron. Measured on a MacBook: a reply raises the microphone's noise floor
about 30x, and Silero calls that speech 58% of the time -- correctly, because it
is speech. Neither energy nor a speech/not-speech classifier can separate the
child from Yarnotron on its own.

What does separate them is that the child is nearer the microphone than the
speaker is. So the detector calibrates against our own echo at the start of each
utterance and then looks for something clearly louder, sustained, and
speech-shaped:

1. a grace window at the start of playback, where the echo level is measured
   and nothing can trigger,
2. a threshold set from that measurement rather than a fixed constant, so it
   follows the volume and the room,
3. a sustained-speech requirement, so a cough or a chair scrape is not an
   interruption.

This is a heuristic, and it assumes the child is closer to the microphone than
the speakers are. Proper acoustic echo cancellation, which subtracts the known
playback signal from the captured one, is the real fix and wants the reference
signal plumbed through. Headphones make the problem disappear entirely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BargeInSettings:
    sample_rate: int = 16_000
    frame_samples: int = 512
    # Nothing can interrupt during this window: it is where the echo level is
    # learned, and it also stops the tail of a previous utterance triggering.
    grace_seconds: float = 0.4
    # Threshold is the 95th percentile of our own echo times this margin.
    # Tuned against a real recording: echo peaks at ~0.10 rms while a child
    # near the microphone sits at ~0.11 median, so the gap is narrow and the
    # margin has to be small.
    echo_percentile: float = 95.0
    # Raised from 1.1 after a real session cut 4 of 22 replies short -- three of
    # them after one sentence, when a reply is normally two, so the child lost
    # the follow-up question. At 1.1 anything a tenth louder than our own echo
    # was an interruption, which room noise clears easily. Tune per room with
    # --barge-in-margin: lower interrupts more eagerly, higher needs a clearly
    # nearer voice. Headphones remove the problem altogether.
    margin: float = 1.4
    # Speech fluctuates, so this is "most of a window", not a run. 8 of the last
    # 12 frames is ~256 ms of speech inside a ~380 ms window. Also raised from
    # 6: a cough or a chair scrape could clear six.
    speech_frames_required: int = 8
    detection_window: int = 12
    speech_probability: float = 0.5
    # Floor, in case the grace window happens to be near-silent.
    fallback_threshold: float = 0.05


class BargeInDetector:
    """Feed it captured frames during playback; it says when to stop talking."""

    def __init__(self, settings: BargeInSettings | None = None, vad=None) -> None:
        self.settings = settings or BargeInSettings()
        self._vad = vad
        self._grace_frames = max(
            1,
            int(self.settings.grace_seconds * self.settings.sample_rate
                / self.settings.frame_samples),
        )
        self.reset()

    def reset(self) -> None:
        """Call at the start of every utterance: the echo level changes with it."""
        self._seen = 0
        self._grace_rms: list[float] = []
        self._threshold: float | None = None
        self._recent: deque[int] = deque(maxlen=self.settings.detection_window)
        self.triggered = False

    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else self.settings.fallback_threshold

    def feed(self, frame: np.ndarray) -> bool:
        """True once the child is judged to be talking over us."""
        if self.triggered:
            return True

        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        self._seen += 1

        if self._seen <= self._grace_frames:
            self._grace_rms.append(rms)
            if self._seen == self._grace_frames:
                echo = float(np.percentile(self._grace_rms, self.settings.echo_percentile))
                self._threshold = max(echo * self.settings.margin,
                                      self.settings.fallback_threshold)
            return False

        louder = rms > self.threshold
        self._recent.append(1 if (louder and self._is_speech(samples)) else 0)
        if sum(self._recent) >= self.settings.speech_frames_required:
            self.triggered = True
        return self.triggered

    def _is_speech(self, samples: np.ndarray) -> bool:
        if self._vad is None:
            return True  # energy only; the caller chose not to supply a VAD
        try:
            probability = float(np.asarray(self._vad(samples)).ravel()[0])
        except Exception:
            return True
        return probability >= self.settings.speech_probability


def load_silero() -> object | None:
    """Silero VAD, if the voice extra is installed. faster-whisper ships it."""
    try:
        from faster_whisper.vad import get_vad_model

        return get_vad_model()
    except Exception:
        return None
