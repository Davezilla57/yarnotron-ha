import json

import pytest

from yarnotron.services.incremental_json import JsonStringFieldReader, SentenceAccumulator

FENCED_DRAFT = (
    '```json\n{\n  "response_text": "Oh, a seal pup! What does Wren do next?",\n'
    '  "story_patches": [{"operation": "add_scene"}],\n  "tutor_notes": []\n}\n```'
)
PLAIN_DRAFT = '{"response_text": "Hello there.", "story_patches": []}'


def feed_all(reader: JsonStringFieldReader, text: str, size: int) -> str:
    return "".join(reader.feed(text[i : i + size]) for i in range(0, len(text), size))


def test_reads_field_from_plain_object():
    reader = JsonStringFieldReader()
    assert reader.feed(PLAIN_DRAFT) == "Hello there."
    assert reader.done


def test_reads_field_through_a_json_fence():
    reader = JsonStringFieldReader()
    assert reader.feed(FENCED_DRAFT) == "Oh, a seal pup! What does Wren do next?"
    assert reader.done


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 13, 64])
def test_survives_any_chunk_boundary(size):
    """The model streams in arbitrary chunks; no split may change the output."""
    reader = JsonStringFieldReader()
    assert feed_all(reader, FENCED_DRAFT, size) == "Oh, a seal pup! What does Wren do next?"
    assert reader.done


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
def test_decodes_escapes_split_across_chunks(size):
    raw = json.dumps({"response_text": 'She said "hi" and\nleft. — fin', "x": 1})
    reader = JsonStringFieldReader()
    assert feed_all(reader, raw, size) == 'She said "hi" and\nleft. — fin'


def test_stops_at_the_closing_quote_and_ignores_later_fields():
    raw = '{"response_text": "Done.", "story_patches": [{"rationale": "not spoken"}]}'
    reader = JsonStringFieldReader()
    assert reader.feed(raw) == "Done."
    assert reader.feed('{"response_text": "again"}') == ""


def test_escaped_quote_is_not_a_terminator():
    raw = json.dumps({"response_text": 'He said \\"stop\\" loudly.'})
    reader = JsonStringFieldReader()
    result = reader.feed(raw)
    assert result.endswith("loudly.")
    assert reader.done


def test_incomplete_stream_yields_partial_text_and_is_not_done():
    reader = JsonStringFieldReader()
    assert reader.feed('{"response_text": "half a sen') == "half a sen"
    assert not reader.done


def test_sentences_release_once_the_next_one_starts():
    """A terminator alone is not a boundary: the next visible character decides.

    That costs one token of delay before the first sentence is released, which
    is the price of not splitting "it... here" or "Mr. Smith".
    """
    accumulator = SentenceAccumulator()
    assert accumulator.feed("Oh, a seal pup! ") == []  # nothing after "!" yet
    assert accumulator.feed("What does Wren do") == ["Oh, a seal pup!"]
    assert accumulator.feed(" next? And then") == ["What does Wren do next?"]
    assert accumulator.flush() == "And then"


def test_sentence_split_is_stable_across_chunk_sizes():
    text = "Wren climbed the stair. Barnacle refused to fly! What happens next?"
    for size in (1, 3, 8, 40):
        accumulator = SentenceAccumulator()
        collected = []
        for i in range(0, len(text), size):
            collected += accumulator.feed(text[i : i + size])
        tail = accumulator.flush()
        if tail:
            collected.append(tail)
        assert collected == [
            "Wren climbed the stair.",
            "Barnacle refused to fly!",
            "What happens next?",
        ], size


def test_does_not_split_on_ellipsis_followed_by_lower_case():
    accumulator = SentenceAccumulator()
    assert accumulator.feed("Wait for it... here it comes! ") == []
    assert accumulator.flush() == "Wait for it... here it comes!"


def test_does_not_split_inside_a_decimal():
    accumulator = SentenceAccumulator()
    assert accumulator.feed("The lamp burned for 3.5 hours that night. Then dawn") == [
        "The lamp burned for 3.5 hours that night."
    ]


def test_splits_when_a_quote_opens_the_next_sentence():
    accumulator = SentenceAccumulator()
    assert accumulator.feed('Wren shouted back. "Hold on!"') == ["Wren shouted back."]


def test_flush_is_idempotent():
    accumulator = SentenceAccumulator()
    accumulator.feed("Tail with no terminator")
    assert accumulator.flush() == "Tail with no terminator"
    assert accumulator.flush() == ""
