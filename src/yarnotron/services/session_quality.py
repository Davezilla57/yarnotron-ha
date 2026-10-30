"""A few counts per session that show whether conversations are going smoothly. Counts only, never words."""

from __future__ import annotations

from yarnotron.agent import code_messages, turn_acts
from yarnotron.models.memory import ConversationTurn


def _is_filler(text: str) -> bool:
    from yarnotron.agent.turn_processor import _is_filler as is_filler

    return is_filler(text)


def summarize(turns: list[ConversationTurn]) -> dict[str, int]:
    """Turn counts for one session. `turns` may be in any order."""
    kinds = lambda t: {n.kind for n in t.tutor_notes}  # noqa: E731
    return {
        "turns": len(turns),
        "filler_turns": sum(1 for t in turns if _is_filler(t.transcript)),
        "held_sentences": sum(1 for t in turns if turn_acts.HOLD_LINE_NOTE in kinds(t)),
        "listening_turns": sum(1 for t in turns if turn_acts.DICTATION_NOTE in kinds(t)),
        "confirmations_asked": sum(1 for t in turns if code_messages.CONFIRM_QUESTION in t.response_text),
        "which_one_asked": sum(1 for t in turns if "I'm not sure which one you mean" in t.response_text),
        "frustration_repairs": sum(1 for t in turns if turn_acts.FRUSTRATION_NOTE in kinds(t)),
        "changes_saved": sum(len(t.applied_story_patches) for t in turns),
        "idle_pauses": sum(1 for t in turns if "idle_loop" in t.safety.categories),
    }
