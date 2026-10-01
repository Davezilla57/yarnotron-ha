"""Compile a bounded view of a story for the prompt.

The prompt used to inline the whole story document, pretty-printed. That works
until a child has been writing for a while: at forty scenes the story is 82% of
the prompt, and prompt evaluation is already the largest single component of
time to first audio at ~1.4 s. Cost should not grow with how much story a child
has written -- if anything the reward for writing more should not be waiting
longer.

What the model actually needs is the story it is currently working in, plus
enough of the rest to stay consistent and to address anything by id:

- **every character, always.** Patches reference characters by id, and the
  editing policy dedupes against this list, so dropping any of them would cause
  duplicates and unresolvable updates. Their notes are truncated instead.
- **recent scenes in full**, because that is where the child is writing.
- **older scenes as id and title only.** Enough to avoid re-adding one and to
  target it with `update_scene`, without carrying its draft text.
- **recent facts**, with a count of how many were left out.

Elisions are explicit in the compiled view. A model that cannot see a scene and
is not told it exists will happily add it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yarnotron.models.story import Story


@dataclass(frozen=True)
class StoryContextLimits:
    full_scenes: int = 6
    listed_scenes: int = 30
    facts: int = 15
    note_chars: int = 140
    draft_chars: int = 400


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def compile_story_context(
    story: Story | None,
    limits: StoryContextLimits | None = None,
) -> dict[str, Any] | None:
    """A bounded, compact dict standing in for the whole story document."""
    if story is None:
        return None
    limits = limits or StoryContextLimits()

    compiled: dict[str, Any] = {
        "id": story.id,
        "title": story.title,
    }
    if story.synopsis:
        compiled["synopsis"] = story.synopsis

    # Every character survives: patches address them by id and the editing
    # policy dedupes against this list.
    compiled["characters"] = [
        {
            key: value
            for key, value in (
                ("id", character.id),
                ("name", character.name),
                ("role", character.role),
                ("traits", character.traits or None),
                ("notes", _clip(character.notes, limits.note_chars)),
            )
            if value
        }
        for character in story.characters
    ]

    ordered = sorted(story.scenes, key=lambda scene: scene.sequence)
    recent = ordered[-limits.full_scenes :] if limits.full_scenes else []
    earlier = ordered[: len(ordered) - len(recent)]

    compiled["scenes"] = [
        {
            key: value
            for key, value in (
                ("id", scene.id),
                ("sequence", scene.sequence),
                ("title", scene.title),
                ("summary", scene.summary),
                ("draft_text", _clip(scene.draft_text, limits.draft_chars)),
            )
            if value is not None
        }
        for scene in recent
    ]

    if earlier:
        listed = earlier[-limits.listed_scenes :]
        compiled["earlier_scenes"] = [
            {"id": scene.id, "sequence": scene.sequence, "title": scene.title}
            for scene in listed
        ]
        if len(earlier) > len(listed):
            compiled["earlier_scenes_not_shown"] = len(earlier) - len(listed)

    facts = story.facts[-limits.facts :] if limits.facts else []
    compiled["facts"] = [{"id": fact.id, "text": fact.text} for fact in facts]
    if len(story.facts) > len(facts):
        compiled["earlier_facts_not_shown"] = len(story.facts) - len(facts)

    return compiled
