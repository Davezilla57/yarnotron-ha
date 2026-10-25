import pytest

from yarnotron.services.pii import REDACTED, redact_contact_details


@pytest.mark.parametrize(
    "text,kind",
    [
        ("I live at 42 Maple Street in Springfield.", "street_address"),
        ("my friend lives at 7 Oak Avenue", "street_address"),
        ("it is at 12 harbor road near me", "street_address"),
        ("we live at forty two maple street", "street_address"),
        ("Email the story to sam.jones@example.com", "email_address"),
        ("sam.jones@example.com", "email_address"),
        ("my email is sam dot jones at gmail dot com", "email_address"),
        ("call me on 555-121-2121", "phone_number"),
        ("five five five one two one two", "phone_number"),
    ],
)
def test_detects_and_removes(text, kind):
    result = redact_contact_details(text)
    assert kind in result.kinds
    assert REDACTED in result.text
    assert "@" not in result.text
    assert not any(ch.isdigit() for ch in result.text)


@pytest.mark.parametrize(
    "text",
    [
        "Wren lives in a lighthouse on a hill by Harbor Road in the story.",
        "Fennick sends a message to the harbor office.",
        "Barnacle's gears are full of salt water now.",
        "She has 3 big dogs on the road.",
        "The dragon has 100 gold coins at the end of the lane.",
        "I hate that name, change it.",
        "She counts one two three four five six",
        "The year is 1999 and the road is long.",
        "meet me at the park",
    ],
)
def test_leaves_ordinary_story_talk_alone(text):
    result = redact_contact_details(text)
    assert not result.found, result
    assert result.text == text


def test_text_keeps_the_rest_of_the_sentence():
    result = redact_contact_details("I live at 42 Maple Street, can the fox visit?")
    assert result.text == f"I live at {REDACTED}, can the fox visit?"
