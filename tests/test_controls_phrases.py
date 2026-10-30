"""Home Assistant's stop and sleep phrases, loaded without Home Assistant (the module is plain Python)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "yarnotron_controls", Path(__file__).parents[1] / "custom_components" / "yarnotron" / "controls.py"
)
controls = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(controls)


def test_stop_phrases_including_misheard_app_names():
    for text in ("Stop Yarnitron.", "stop yarn a tron", "Stop the story", "end the story", "stop"):
        assert controls.classify(text) == controls.ControlIntent.STOP_SESSION, text


def test_goodbyes_put_it_to_sleep_without_ending_the_story():
    for text in ("Bye.", "bye bye", "Bye Yarnotron", "see you later"):
        assert controls.classify(text) == controls.ControlIntent.SLEEP, text


def test_story_lines_are_not_commands():
    for text in ("The dragon says bye to the fox", "The knight did not stop", "Stop the dragon from leaving"):
        assert controls.classify(text) is None, text
