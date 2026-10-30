from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, TypeVar

from yarnotron.models.agent import TurnIntent
from yarnotron.models.story import Character, Scene, Story, StoryPatch, StoryPatchOperation


T = TypeVar("T")

DESTRUCTIVE_OPERATIONS = {
    StoryPatchOperation.REMOVE_CHARACTER,
    StoryPatchOperation.REMOVE_SCENE,
    StoryPatchOperation.REMOVE_FACT,
}
UPDATE_OPERATIONS = {
    StoryPatchOperation.UPDATE_TITLE,
    StoryPatchOperation.UPDATE_SYNOPSIS,
    StoryPatchOperation.UPDATE_CHARACTER,
    StoryPatchOperation.UPDATE_SCENE,
}
EXPLICIT_EDIT_WORDS = {
    "change",
    "delete",
    "erase",
    "remove",
    "rename",
    "replace",
    "undo",
    "update",
}


@dataclass
class StoryEditingPolicyResult:
    patches: list[StoryPatch]
    confirmation_notes: list[str] = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        return any(patch.child_confirmation_required for patch in self.patches)


class StoryEditingPolicy:
    def review(
        self,
        patches: list[StoryPatch],
        story: Story | None,
        transcript: str,
        intent: TurnIntent,
    ) -> StoryEditingPolicyResult:
        reviewed: list[StoryPatch] = []
        notes: list[str] = []
        explicit_edit = _contains_explicit_edit(transcript) or intent == TurnIntent.EDIT_STORY

        for patch in patches:
            normalized = self._merge_duplicate_add(patch, story)
            normalized = self._preserve_child_wording(normalized, transcript, intent)
            normalized, note = self._apply_confirmation_rules(
                normalized,
                story,
                explicit_edit=explicit_edit,
            )
            if note is not None:
                notes.append(note)
            reviewed.append(normalized)

        return StoryEditingPolicyResult(patches=reviewed, confirmation_notes=notes)

    def _merge_duplicate_add(self, patch: StoryPatch, story: Story | None) -> StoryPatch:
        if story is None:
            return patch

        if patch.operation == StoryPatchOperation.ADD_CHARACTER:
            name = patch.data.get("name")
            if not isinstance(name, str):
                return patch
            match = _best_character_match(name, story.characters)
            if match is None:
                return patch
            data = _merge_character_data(match, patch.data)
            return patch.model_copy(
                update={
                    "operation": StoryPatchOperation.UPDATE_CHARACTER,
                    "data": data,
                    "rationale": _append_rationale(
                        patch.rationale,
                        f"Merged duplicate character '{name}' into existing character '{match.name}'.",
                    ),
                }
            )

        if patch.operation == StoryPatchOperation.ADD_SCENE:
            title = patch.data.get("title")
            if not isinstance(title, str):
                return patch
            match = _best_scene_match(title, story.scenes)
            if match is None:
                return patch
            data = _merge_scene_data(match, patch.data)
            return patch.model_copy(
                update={
                    "operation": StoryPatchOperation.UPDATE_SCENE,
                    "data": data,
                    "rationale": _append_rationale(
                        patch.rationale,
                        f"Merged duplicate scene '{title}' into existing scene '{match.title}'.",
                    ),
                }
            )

        return patch

    def _preserve_child_wording(
        self,
        patch: StoryPatch,
        transcript: str,
        intent: TurnIntent,
    ) -> StoryPatch:
        if intent not in {TurnIntent.CONTINUE_STORY, TurnIntent.EDIT_STORY}:
            return patch
        if patch.operation not in {StoryPatchOperation.ADD_SCENE, StoryPatchOperation.UPDATE_SCENE}:
            return patch
        if not _looks_like_story_wording(transcript):
            return patch

        data = dict(patch.data)
        existing_draft = data.get("draft_text")
        if isinstance(existing_draft, str) and _normalized(transcript) in _normalized(existing_draft):
            return patch
        if isinstance(existing_draft, str) and existing_draft.strip():
            data["draft_text"] = f"{existing_draft.strip()}\n\n{transcript.strip()}"
        else:
            data["draft_text"] = transcript.strip()

        return patch.model_copy(
            update={
                "data": data,
                "rationale": _append_rationale(
                    patch.rationale,
                    "Preserved the child's original spoken wording in draft_text.",
                ),
            }
        )

    def _apply_confirmation_rules(
        self,
        patch: StoryPatch,
        story: Story | None,
        *,
        explicit_edit: bool,
    ) -> tuple[StoryPatch, str | None]:
        if patch.operation in DESTRUCTIVE_OPERATIONS:
            return _require_confirmation(
                patch,
                "Destructive edits need child confirmation before they are applied.",
            )

        # The child is only asked about the things that are hard to take back: removals (above), renames, and
        # clearing existing wording (below). Adding to or adjusting something does not need a "should I?".
        if _is_rename(patch, story):
            return _require_confirmation(
                patch,
                "Renaming something that already has a name needs child confirmation.",
            )

        if _clears_existing_story_text(patch, story):
            return _require_confirmation(
                patch,
                "The edit would clear existing story wording.",
            )

        return patch, None


def _is_rename(patch: StoryPatch, story: Story | None) -> bool:
    """True when the patch gives an already-named story, character or scene a different name.

    Naming a story that is still untitled, or a character that had no name, is not a rename.
    """
    data = patch.data
    if patch.operation == StoryPatchOperation.UPDATE_TITLE:
        current = (story.title if story else "") or ""
        new = str(data.get("title", ""))
        return bool(current.strip()) and current.strip().casefold() != "untitled story" and _normalized(new) != _normalized(current)
    if story is None:
        return False
    if patch.operation == StoryPatchOperation.UPDATE_CHARACTER and "name" in data:
        existing = next((c for c in story.characters if c.id == data.get("id")), None)
        return existing is not None and bool(existing.name) and _normalized(str(data["name"])) != _normalized(existing.name)
    if patch.operation == StoryPatchOperation.UPDATE_SCENE and "title" in data:
        existing = next((sc for sc in story.scenes if sc.id == data.get("id")), None)
        return existing is not None and bool(existing.title) and _normalized(str(data["title"])) != _normalized(existing.title)
    return False


def is_undo_request(transcript: str) -> bool:
    text = transcript.lower()
    return "undo last" in text or "undo that" in text or "go back" in text or "revert" in text


def _require_confirmation(patch: StoryPatch, note: str) -> tuple[StoryPatch, str]:
    return (
        patch.model_copy(
            update={
                "child_confirmation_required": True,
                "rationale": _append_rationale(patch.rationale, note),
            }
        ),
        note,
    )


def _is_duplicate_merge(patch: StoryPatch) -> bool:
    return bool(patch.rationale and "Merged duplicate" in patch.rationale)


def _contains_explicit_edit(transcript: str) -> bool:
    words = set(re.findall(r"[a-z']+", transcript.lower()))
    return any(word in words for word in EXPLICIT_EDIT_WORDS)


def _best_character_match(name: str, characters: list[Character]) -> Character | None:
    return _best_match(name, characters, lambda character: character.name)


def _best_scene_match(title: str, scenes: list[Scene]) -> Scene | None:
    return _best_match(title, scenes, lambda scene: scene.title)


def _best_match(target: str, items: list[T], name_getter: Callable[[T], str]) -> T | None:
    normalized_target = _normalized(target)
    if not normalized_target:
        return None
    best_item = None
    best_score = 0.0
    for item in items:
        normalized_name = _normalized(name_getter(item))
        score = SequenceMatcher(None, normalized_target, normalized_name).ratio()
        if normalized_target in normalized_name or normalized_name in normalized_target:
            score = max(score, 0.92)
        if score > best_score:
            best_item = item
            best_score = score
    return best_item if best_score >= 0.86 else None


def _merge_character_data(character: Character, incoming: dict[str, object]) -> dict[str, object]:
    traits = list(character.traits)
    incoming_traits = incoming.get("traits")
    if isinstance(incoming_traits, list):
        for trait in incoming_traits:
            trait_text = str(trait)
            if trait_text and trait_text not in traits:
                traits.append(trait_text)

    return {
        "id": character.id,
        "name": str(incoming.get("name") or character.name),
        "role": incoming.get("role") or character.role,
        "traits": traits,
        "notes": _merge_text(character.notes, incoming.get("notes")),
    }


def _merge_scene_data(scene: Scene, incoming: dict[str, object]) -> dict[str, object]:
    sequence = incoming.get("sequence", scene.sequence)
    return {
        "id": scene.id,
        "title": str(incoming.get("title") or scene.title),
        "summary": _merge_text(scene.summary, incoming.get("summary")) or scene.summary,
        "draft_text": _merge_text(scene.draft_text, incoming.get("draft_text")),
        "sequence": _coerce_sequence(sequence, scene.sequence),
    }


def _merge_text(existing: str | None, incoming: object) -> str | None:
    incoming_text = incoming.strip() if isinstance(incoming, str) else ""
    existing_text = existing.strip() if isinstance(existing, str) else ""
    if not incoming_text:
        return existing_text or None
    if not existing_text:
        return incoming_text
    if _normalized(incoming_text) in _normalized(existing_text):
        return existing_text
    return f"{existing_text}\n\n{incoming_text}"


def _coerce_sequence(value: object, default: int) -> int:
    try:
        return int(value) if isinstance(value, int | str) else default
    except ValueError:
        return default


def _clears_existing_story_text(patch: StoryPatch, story: Story | None) -> bool:
    if story is None:
        return False
    if patch.operation == StoryPatchOperation.UPDATE_SYNOPSIS:
        return bool(story.synopsis) and patch.data.get("synopsis") in {None, ""}
    if patch.operation == StoryPatchOperation.UPDATE_SCENE:
        scene_id = patch.data.get("id")
        scene = next((item for item in story.scenes if item.id == scene_id), None)
        if scene is None:
            return False
        clears_summary = bool(scene.summary) and patch.data.get("summary") in {None, ""}
        clears_draft = bool(scene.draft_text) and patch.data.get("draft_text") in {None, ""}
        return clears_summary or clears_draft
    return False


def _looks_like_story_wording(transcript: str) -> bool:
    text = transcript.strip()
    if len(text) < 20:
        return False
    lowered = text.lower()
    if lowered.startswith(("change ", "delete ", "remove ", "rename ", "undo ")):
        return False
    return True


def _append_rationale(existing: str | None, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing} {addition}"


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
