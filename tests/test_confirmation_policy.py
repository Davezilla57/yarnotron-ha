"""The child is asked "should I make it?" only for removals, renames and clearing wording."""

from yarnotron.agent.editing_policy import StoryEditingPolicy
from yarnotron.models.agent import TurnIntent
from yarnotron.models.story import Character, Scene, Story, StoryPatch, StoryPatchOperation as Op

POLICY = StoryEditingPolicy()


def story(title="The Fox Story"):
    return Story(title=title, characters=[Character(id="character_pip", name="Pip", role="fox")],
                 scenes=[Scene(id="scene_a", title="The Den", summary="a den")])


def confirm(op, data, st=None, transcript="the fox is brave"):
    patch = StoryPatch(operation=op, data=data, rationale="t")
    out = POLICY.review([patch], st or story(), transcript, TurnIntent.CONTINUE_STORY)
    return out.patches[0].child_confirmation_required


def test_adjusting_or_adding_does_not_ask():
    assert not confirm(Op.UPDATE_CHARACTER, {"id": "character_pip", "traits": ["brave"]})
    assert not confirm(Op.UPDATE_CHARACTER, {"id": "character_pip", "role": "a fox in a hat"})
    assert not confirm(Op.UPDATE_SCENE, {"id": "scene_a", "summary": "a cosy den"})
    assert not confirm(Op.UPDATE_SYNOPSIS, {"synopsis": "A fox finds a lantern."})
    assert not confirm(Op.ADD_FACT, {"text": "It is night."})


def test_removals_and_renames_ask_even_when_the_child_said_so_plainly():
    assert confirm(Op.REMOVE_CHARACTER, {"id": "character_pip"})
    assert confirm(Op.UPDATE_CHARACTER, {"id": "character_pip", "name": "Pop"}, transcript="rename Pip to Pop")
    assert confirm(Op.UPDATE_TITLE, {"title": "The Lantern"}, transcript="rename this to The Lantern")
    assert confirm(Op.UPDATE_SCENE, {"id": "scene_a", "title": "The Cave"})


def test_first_naming_of_an_untitled_story_does_not_ask():
    assert not confirm(Op.UPDATE_TITLE, {"title": "The Elephant Walk"}, st=story(title="Untitled Story"))
    assert not confirm(Op.UPDATE_CHARACTER, {"id": "character_pip", "name": "Pip"})
