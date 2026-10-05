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
    LAUNCH = "launch"
    SHUTDOWN = "shutdown"
    STATUS = "status"


# These act on the machines, not the story. They are matched on the whole
# utterance too, and always name Yarnotron (bare "status" aside), so a story line
# like "the dragon starts the engine" never triggers one.
POWER_INTENTS = frozenset({ControlIntent.LAUNCH, ControlIntent.SHUTDOWN, ControlIntent.STATUS})


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


_LAUNCH = {
    "launch yarnotron",
    "start yarnotron",
    "boot yarnotron",
    "boot up yarnotron",
    "start up yarnotron",
    "wake up yarnotron",
    "turn on yarnotron",
    "power on yarnotron",
    "power up yarnotron",
    "fire up yarnotron",
    "bring up yarnotron",
}
_SHUTDOWN = {
    "shut down yarnotron",
    "shutdown yarnotron",
    "shut yarnotron down",
    "turn off yarnotron",
    "turn yarnotron off",
    "power down yarnotron",
    "power off yarnotron",
    "close yarnotron",
    "quit yarnotron",
    "exit yarnotron",
}
_STATUS = {
    "yarnotron status",
    "status of yarnotron",
    "status",
    "system status",
    "check yarnotron",
    "check status",
    "check yarnotron status",
    "is yarnotron running",
    "is yarnotron up",
    "is yarnotron ok",
    "what is running",
    "whats running",
    "yarnotron health",
    "health check",
    "give me a status report",
    "status report",
}


def normalize(text: str) -> str:
    folded = text.casefold().replace("'", "").replace("’", "")
    folded = re.sub(r"[^a-z0-9\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def classify(text: str) -> ControlIntent | None:
    normalized = normalize(text)
    if not normalized:
        return None
    # "please launch yarnotron", "launch yarnotron please", "yarnotron, launch yarnotron"
    stripped = re.sub(r"^yarnotron (?=(?:launch|start|shut|turn|power|check|status|close|quit)\b)", "", normalized)
    stripped = re.sub(r"^please |( please| now)+$", "", stripped)
    if stripped in _LAUNCH:
        return ControlIntent.LAUNCH
    if stripped in _SHUTDOWN:
        return ControlIntent.SHUTDOWN
    if stripped in _STATUS:
        return ControlIntent.STATUS
    if normalized in _HELP:
        return ControlIntent.HELP
    if normalized in _REPEAT:
        return ControlIntent.REPEAT
    if normalized in _SLEEP:
        return ControlIntent.SLEEP
    if normalized in _STOP:
        return ControlIntent.STOP_SESSION
    return None


_YES = {
    "yes",
    "yes shut down",
    "yes shutdown",
    "yes please",
    "yes do it",
    "yeah",
    "yep",
    "confirm",
    "confirmed",
    "do it",
    "go ahead",
}
_NO = {"no", "no thanks", "cancel", "never mind", "nevermind", "dont", "do not", "leave it", "stop"}


def is_confirmation(text: str) -> bool | None:
    """True for a yes to a pending shutdown, False for a no, None for anything else."""
    normalized = normalize(text)
    if normalized in _YES or classify(text) is ControlIntent.SHUTDOWN:
        return True
    if normalized in _NO or normalized == "never mind":
        return False
    return None
