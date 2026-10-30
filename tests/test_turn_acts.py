import pytest

from yarnotron.agent import turn_acts as acts


@pytest.mark.parametrize("text", ["No, let me tell you about it", "Can I tell you about the nation's flag?", "Just let me tell you about its flag!", "no let me tell you."])
def test_hand_over_phrases(text):
    assert acts.wants_to_tell(text)


@pytest.mark.parametrize("text", ["She cramps back into a rocket and", "What happens next is the", "The Space Nations name is the", "Then she goes to...", "First she finds, the first world or planet that she finds is"])
def test_dangling_fragments(text):
    assert acts.is_dangling(text)


@pytest.mark.parametrize("text", ["The fox ran home.", "What happens next?", "She flew to the moon!", "Yes", "It is a star.", "", "undo that", "I like it", "I think so", "do it"])
def test_complete_lines_are_not_fragments(text):
    assert not acts.is_dangling(text)


@pytest.mark.parametrize("text", ["Why can't you do anything?", "No, that's not what I said", "It's not your business", "you're stupid"])
def test_frustration(text):
    assert acts.is_frustrated(text)


def test_new_story_delete_story_and_parent_address():
    assert acts.asks_new_story("Can we create a new story?") and acts.asks_new_story("Let's start a brand new story")
    assert not acts.asks_new_story("The new story began")
    assert acts.asks_delete_story("Delete that story.") and not acts.asks_delete_story("Delete that character")
    assert acts.addresses_parent("Dad, can you...") and not acts.addresses_parent("The dad bear walked in")


def test_leaving_dictation():
    assert acts.leaves_dictation("What happens next?") and acts.leaves_dictation("you tell me")
    assert not acts.leaves_dictation("The flag has fifty stars")
