from yarnotron.models.story import Story, StoryPatch, StoryPatchOperation
from yarnotron.services.story_store import StoryPatchApplier, story_id_from_title


def test_story_id_from_title_is_human_readable() -> None:
    assert story_id_from_title("Clock Tower Mystery") == "story_clock_tower_mystery"
    assert story_id_from_title("  Outer Space: All the Planets!  ") == (
        "story_outer_space_all_the_planets"
    )
    assert story_id_from_title("!!!") == "story_untitled_story"


def test_create_and_patch_story_title() -> None:
    applier = StoryPatchApplier()
    story = Story(title="Clock Tower Mystery")

    updated = applier.apply(
        story,
        StoryPatch(
            operation=StoryPatchOperation.UPDATE_TITLE,
            data={"title": "The Clock Tower Inventor"},
        ),
    )

    assert updated.title == "The Clock Tower Inventor"


def test_add_scene_sorts_by_sequence() -> None:
    applier = StoryPatchApplier()
    story = Story(title="Sky Bridge")

    story = applier.apply(
        story,
        StoryPatch(
            operation=StoryPatchOperation.ADD_SCENE,
            data={"title": "Ending", "summary": "They cross the bridge.", "sequence": 2},
        ),
    )
    updated = applier.apply(
        story,
        StoryPatch(
            operation=StoryPatchOperation.ADD_SCENE,
            data={"title": "Beginning", "summary": "They find the map.", "sequence": 1},
        ),
    )

    assert [scene.title for scene in updated.scenes] == ["Beginning", "Ending"]
