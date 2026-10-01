import re

from yarnotron.agent.prompts import build_story_turn_prompt
from yarnotron.models.agent import AgentTurnRequest, TurnIntent

GOOD_EXAMPLES_HEADING = "Good, each two sentences ending on the question"


def story_prompt(transcript: str = "Milo finds a tiny silver map inside the clock.") -> str:
    return build_story_turn_prompt(
        AgentTurnRequest(story_id="story_test", transcript=transcript),
        TurnIntent.CONTINUE_STORY,
    )


def test_story_turn_prompt_includes_playful_interactive_voice_contract() -> None:
    prompt = story_prompt().casefold()

    assert "acknowledging the specific thing the child just said" in prompt
    assert '"yes, and" energy' in prompt
    assert "they stay author" in prompt
    assert "end on one short open question" in prompt
    assert "two closed choices only when" in prompt
    assert "never generic" in prompt


def test_story_turn_prompt_bounds_the_spoken_length() -> None:
    """Replies are spoken, so length is time the child spends listening.

    Before this contract the corpus averaged 7.5 s of speech per reply and 44 of
    56 replies were over the 5 s target.
    """
    prompt = story_prompt()

    assert "Two short sentences" in prompt
    assert "about 110 characters" in prompt
    # The one-question rule, which is what actually stops replies rambling on.
    assert "never a second question, never a sentence after it" in prompt
    assert "Too long" in prompt


def test_every_good_example_is_short_enough_to_be_the_model_target() -> None:
    """The examples teach length more strongly than the instruction does."""
    prompt = story_prompt()
    good = prompt.split(GOOD_EXAMPLES_HEADING)[1].split("Too long")[0]
    examples = re.findall(r'^- "(.+)"$', good, flags=re.MULTILINE)

    assert len(examples) >= 2
    for example in examples:
        assert len(example) <= 110, f"example is {len(example)} chars: {example}"
        assert example.rstrip().endswith("?"), f"should end on the question: {example}"


def test_the_counter_example_is_actually_too_long() -> None:
    """A bad example that fits the target teaches the wrong thing."""
    prompt = story_prompt()
    too_long = prompt.split("Too long")[1].split("Return only valid JSON")[0]
    spoken = " ".join(re.findall(r'"(.+?)"', too_long, flags=re.DOTALL)).strip()

    assert len(spoken) > 110


def test_every_patch_operation_is_documented() -> None:
    """The model cannot emit an operation whose fields the prompt never named.

    The operation list was compressed onto shared lines to save prompt tokens,
    which is exactly the kind of edit that can silently drop one.
    """
    from yarnotron.models.story import StoryPatchOperation

    prompt = story_prompt()

    for operation in StoryPatchOperation:
        assert operation.value in prompt, f"{operation.value} is not in the prompt"
