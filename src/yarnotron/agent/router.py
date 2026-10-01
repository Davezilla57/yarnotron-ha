from yarnotron.models.agent import TurnIntent
from yarnotron.agent.editing_policy import is_undo_request


class IntentRouter:
    def route(self, transcript: str) -> TurnIntent:
        text = transcript.lower()
        normalized = _normalize_router_text(transcript)

        if _is_unclear_story_request(normalized):
            return TurnIntent.UNKNOWN

        if is_undo_request(text):
            return TurnIntent.UNDO_LAST_CHANGE
        if "what does" in text or "define" in text or "definition" in text:
            return TurnIntent.ASK_DEFINITION
        if "grammar" in text or "spell" in text or "sentence" in text:
            return TurnIntent.ASK_GRAMMAR
        if "picture" in text or "image" in text or "draw" in text:
            return TurnIntent.CREATE_IMAGE
        if "change" in text or "remove" in text or "add" in text or "rename" in text:
            return TurnIntent.EDIT_STORY
        if "parent" in text or "dad" in text or "mom" in text:
            return TurnIntent.PARENT_COMMAND
        return TurnIntent.CONTINUE_STORY


def _normalize_router_text(transcript: str) -> str:
    return " ".join(
        transcript.casefold()
        .replace("'", "")
        .replace("?", " ")
        .replace("!", " ")
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )


def _is_unclear_story_request(normalized: str) -> bool:
    return normalized in {
        "huh",
        "uh",
        "um",
        "umm",
        "what",
        "wait what",
        "i dont know",
        "idk",
        "not sure",
        "no idea",
        "i have no idea",
        "never mind",
        "nevermind",
        "nothing",
        "whatever",
        "okay",
        "ok",
        "yes",
        "no",
        "maybe",
    }
