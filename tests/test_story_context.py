from __future__ import annotations

import json

from yarnotron.agent.prompts import build_story_turn_prompt
from yarnotron.agent.story_context import StoryContextLimits, compile_story_context
from yarnotron.models.agent import AgentTurnRequest, TurnIntent
from yarnotron.models.story import Character, Scene, Story, StoryFact


def story_of(scenes: int, characters: int = 3, facts: int = 4) -> Story:
    return Story(
        title="The Lighthouse at Sable Point",
        synopsis="Wren keeps the last lighthouse burning.",
        characters=[
            Character(name=f"Character {i}", role="companion", traits=["brave"],
                      notes="A note " * 40)
            for i in range(characters)
        ],
        scenes=[
            Scene(title=f"Scene {i}", summary=f"Summary {i}", sequence=i,
                  draft_text="The child's own words. " * 40)
            for i in range(scenes)
        ],
        facts=[StoryFact(text=f"Fact {i}") for i in range(facts)],
    )


def test_no_story_compiles_to_nothing():
    assert compile_story_context(None) is None


def test_a_small_story_is_carried_whole():
    story = story_of(scenes=3, facts=2)
    compiled = compile_story_context(story)

    assert [s["title"] for s in compiled["scenes"]] == ["Scene 0", "Scene 1", "Scene 2"]
    assert "earlier_scenes" not in compiled
    assert len(compiled["facts"]) == 2
    assert "earlier_facts_not_shown" not in compiled


def test_every_character_survives_however_long_the_story_gets():
    """Patches address characters by id and the editing policy dedupes on this
    list, so a dropped character means duplicates and unresolvable updates."""
    story = story_of(scenes=80, characters=12)
    compiled = compile_story_context(story)

    assert len(compiled["characters"]) == 12
    assert all(character["id"] for character in compiled["characters"])


def test_recent_scenes_stay_whole_and_older_ones_keep_their_ids():
    story = story_of(scenes=20)
    compiled = compile_story_context(story, StoryContextLimits(full_scenes=4))

    assert [s["title"] for s in compiled["scenes"]] == [
        "Scene 16", "Scene 17", "Scene 18", "Scene 19",
    ]
    assert all("draft_text" in s for s in compiled["scenes"])
    # Older scenes are still addressable, just not carried in full.
    earlier = compiled["earlier_scenes"]
    assert len(earlier) == 16
    assert all(set(s) == {"id", "sequence", "title"} for s in earlier)


def test_every_scene_id_is_still_reachable():
    """The model must be able to target any scene with update_scene."""
    story = story_of(scenes=25)
    compiled = compile_story_context(story, StoryContextLimits(full_scenes=5))

    reachable = {s["id"] for s in compiled["scenes"]}
    reachable |= {s["id"] for s in compiled.get("earlier_scenes", [])}
    assert reachable == {scene.id for scene in story.scenes}


def test_elisions_are_declared_rather_than_silent():
    """A model that cannot see a scene and is not told will re-add it."""
    story = story_of(scenes=60, facts=40)
    compiled = compile_story_context(
        story, StoryContextLimits(full_scenes=3, listed_scenes=10, facts=5)
    )

    assert compiled["earlier_scenes_not_shown"] == 60 - 3 - 10
    assert compiled["earlier_facts_not_shown"] == 40 - 5


def test_long_notes_and_drafts_are_clipped_not_dropped():
    story = story_of(scenes=1, characters=1)
    compiled = compile_story_context(story, StoryContextLimits(note_chars=40, draft_chars=50))

    assert len(compiled["characters"][0]["notes"]) <= 41
    assert compiled["characters"][0]["notes"].endswith("…")
    assert len(compiled["scenes"][0]["draft_text"]) <= 51


def test_facts_keep_the_most_recent():
    story = story_of(scenes=1, facts=10)
    compiled = compile_story_context(story, StoryContextLimits(facts=3))
    assert [f["text"] for f in compiled["facts"]] == ["Fact 7", "Fact 8", "Fact 9"]


def test_scenes_are_ordered_by_sequence_not_insertion():
    story = Story(
        title="Out of order",
        scenes=[
            Scene(title="Third", summary="c", sequence=3),
            Scene(title="First", summary="a", sequence=1),
            Scene(title="Second", summary="b", sequence=2),
        ],
    )
    compiled = compile_story_context(story, StoryContextLimits(full_scenes=2))
    assert [s["title"] for s in compiled["scenes"]] == ["Second", "Third"]
    assert [s["title"] for s in compiled["earlier_scenes"]] == ["First"]


def test_prompt_growth_is_bounded_by_story_length():
    """The whole point: writing more story must not cost more prompt forever."""
    request = AgentTurnRequest(transcript="What happens next?")

    def prompt_chars(scenes: int) -> int:
        return len(
            build_story_turn_prompt(
                request, TurnIntent.CONTINUE_STORY, story=story_of(scenes), relevant_memories=[]
            )
        )

    small, large, huge = prompt_chars(5), prompt_chars(40), prompt_chars(200)
    # Growth from 40 to 200 scenes -- five times the story -- must be modest,
    # because only ids and titles accumulate past the window.
    assert huge < large * 2.5
    # And the whole document version would have been far larger than this.
    whole = len(json.dumps(story_of(200).model_dump(mode="json"), indent=2))
    assert huge < whole / 2
    assert small < large


def test_prompt_is_minified():
    story = story_of(scenes=3)
    prompt = build_story_turn_prompt(
        AgentTurnRequest(transcript="hi"), TurnIntent.CONTINUE_STORY,
        story=story, relevant_memories=[],
    )
    # Compact separators, not indent=2.
    assert '{"id":' in prompt
    assert '\n  "id"' not in prompt


def test_prompt_warns_the_model_the_view_is_partial():
    story = story_of(scenes=40)
    prompt = build_story_turn_prompt(
        AgentTurnRequest(transcript="hi"), TurnIntent.CONTINUE_STORY,
        story=story, relevant_memories=[],
    )
    assert "bounded view" in prompt
    assert "Never re-add" in prompt
