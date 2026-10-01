"""Spoken controls handled locally, without a round trip to the backend.

Ported from `yarnotron.voice.loop.classify_control_intent`. The wake-word,
pause and resume states of the original voice loop do not exist here: the
satellite owns the wake word, and "pause" is just not speaking until the next
wake. Matching is on the whole utterance, exactly as before, so "stop the
dragon" is a story turn and "stop" is a control.
"""

from __future__ import annotations

import re
from enum import StrEnum


class ControlIntent(StrEnum):
    HELP = "help"
    REPEAT = "repeat"
    SLEEP = "sleep"
    STOP_SESSION = "stop_session"


_HELP = {
    "help",
    "help me",
    "what can i say",
    "what can you do",
    "how does this work",
    "tell me what i can say",
}
_REPEAT = {
    "repeat",
    "repeat that",
    "say that again",
    "say it again",
    "what did you say",
}
# "pause" and its phrases are folded in with sleep: with a satellite there is
# nothing to resume, the child simply wakes it again.
_SLEEP = {
    "sleep",
    "go to sleep",
    "stop listening",
    "thats all",
    "that is all",
    "never mind",
    "pause",
    "pause yarnotron",
    "hold on",
    "wait a second",
}
_STOP = {
    "stop",
    "stop session",
    "end session",
    "quit",
    "exit",
    "goodbye",
}


def normalize(text: str) -> str:
    folded = text.casefold().replace("'", "").replace("’", "")
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def classify(text: str) -> ControlIntent | None:
    normalized = normalize(text)
    if not normalized:
        return None
    if normalized in _HELP:
        return ControlIntent.HELP
    if normalized in _REPEAT:
        return ControlIntent.REPEAT
    if normalized in _SLEEP:
        return ControlIntent.SLEEP
    if normalized in _STOP:
        return ControlIntent.STOP_SESSION
    return None
