"""Things a child says that are about the conversation, not the story. Code answers these, not the model.

Each detector is a plain function over the words, so it is cheap, testable, and cannot be talked into anything.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------------------------------- words


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]+", " ", text.casefold().replace("’", "'")).split())


LISTEN_REPLIES = ("Okay, I'm listening.", "Go ahead, I'm listening.", "Mm-hm, tell me.")
HOLD_REPLIES = ("Mm-hm, go on.", "I'm listening.", "Okay, keep going.")
HOLD_LINE_NOTE = "fragment"
DICTATION_NOTE = "dictation"
FRUSTRATION_NOTE = "frustration"

REPAIR_LINE = "I'm sorry, I'm not getting that right. Tell me what you'd like, and I'll just listen."
GROWN_UP_LINE = "That sounds like one for your grown-up. Go ahead and ask them, and I'll be here when you're ready."
STORY_DELETE_LINE = "Only a grown-up can delete a story. We can start a new one, or keep going with this one."
NEW_STORY_LINE = "Okay, a brand new story. What should we call it?"

# --------------------------------------------------------------------------------------------- detectors

_HANDOVER = re.compile(
    r"\b(let me (?:tell|say|finish|explain)|can i (?:tell|say|finish|explain)|i(?:'ll| will| want to) (?:tell|say) you|"
    r"let me do it|i(?:'ll| will) tell you|just let me)\b"
)
_DANGLING_LAST = frozenset(
    "and but or because the a an to of with from in on at for by is are was were its his their who "
    "which where when as if into onto about like than also while until after before my your our".split()
)
_FRUSTRATED = re.compile(
    r"\b(why can'?t you|you(?:'re| are) not listening|you don'?t (?:listen|get it|understand)|that'?s not what i (?:said|meant)|"
    r"not your business|you(?:'re| are) (?:stupid|dumb|silly|bad|wrong)|stop it|i (?:said|told you)|no no|"
    r"that'?s (?:wrong|not right)|you can'?t do anything|you never)\b"
)
_NEW_STORY = re.compile(r"\b(?:create|start|make|begin|write|do|have)\b (?:me )?(?:a |another |one )?(?:brand )?(?:new|different|another) (?:story|one)\b")
_DELETE_STORY = re.compile(r"\b(?:delete|remove|erase|get rid of|throw away)\b (?:this|that|the|my) (?:whole )?story\b")
_ADDRESSED_PARENT = re.compile(r"^(?:hey |hi |oh |um |uh )?(?:dad|daddy|mom|mommy|mama|papa|grandma|grandpa)\b")
_EXIT_DICTATION = re.compile(r"\b(what happens next|your turn|you tell|you go|what do you think|you decide|ask me)\b")


def wants_to_tell(transcript: str) -> bool:
    return bool(_HANDOVER.search(_plain(transcript)))


def is_dangling(transcript: str) -> bool:
    """True when the child's last words stop mid-thought ("...a rocket and", "the name is the")."""
    text = transcript.strip()
    if not text or text.endswith(("?", "!")):
        return False
    if text.endswith(("...", "…", "-", "—", ",")):
        return True
    words = _plain(text).split()
    return bool(words) and words[-1] in _DANGLING_LAST and not text.endswith(".")


def is_frustrated(transcript: str) -> bool:
    return bool(_FRUSTRATED.search(_plain(transcript)))


def asks_new_story(transcript: str) -> bool:
    return bool(_NEW_STORY.search(_plain(transcript)))


def asks_delete_story(transcript: str) -> bool:
    return bool(_DELETE_STORY.search(_plain(transcript)))


def addresses_parent(transcript: str) -> bool:
    return bool(_ADDRESSED_PARENT.search(_plain(transcript)))


def leaves_dictation(transcript: str) -> bool:
    text = transcript.strip()
    return text.endswith("?") or bool(_EXIT_DICTATION.search(_plain(text)))


def pick(options: tuple[str, ...], turn_number: int) -> str:
    return options[turn_number % len(options)]
