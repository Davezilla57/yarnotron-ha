"""Words that code says instead of the model. Fixed, short, and approved by a parent.

Both worded lines were approved by the parent on 2026-10-25. NOT_HEARD, IDLE_PAUSE and `which_one`, and the lines in `turn_acts`, were approved on 2026-10-29. They must match `spec/agent.yaml` (tests check this).
"""

WORRIED_CHILD_SCRIPT = (
    "Thank you for telling me. That sounds like something a grown-up should know about, "
    "so please go and tell one now. I'm going to stop our story for a little while."
)

SCENE_OFFER_QUESTION = "Want me to save that as its own scene?"
CONFIRM_QUESTION = "That's a big change. Should I make it?"
UNDO_DONE = "No problem. I put the story back the way it was before the last change. What would you like to try instead?"
UNDO_NOTHING = "I do not see a recent story change to undo yet. What tiny change should we try next?"
PICTURE_SAVED = "I saved that picture idea for a grown-up to approve, and we can keep writing while it waits."
SAVE_FAILED = "I could not save that just now, but we can keep going with the story."
UNCLEAR_TARGET = "I'm not sure which one you mean. Can you tell me its name again?"
NOT_HEARD = "I didn't catch that. Can you say it again?"
IDLE_PAUSE = "I'll take a little rest now. Ask a grown-up to start again whenever you want more story."


def which_one(names: list[str]) -> str:
    """Ask which one, offering the characters by name when there are few enough to say aloud."""
    if 1 <= len(names) <= 4:
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"I'm not sure which one you mean. Is it {listed}?"
    return UNCLEAR_TARGET
