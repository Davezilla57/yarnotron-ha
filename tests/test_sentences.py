from yarnotron.services.incremental_json import SentenceAccumulator
from yarnotron.services.sentences import drop_trailing_question, split_sentences


def test_mrs_is_not_a_sentence_end():
    text = "The rocket lifts off. What does Mrs. Strumson do next?"
    assert split_sentences(text) == ["The rocket lifts off.", "What does Mrs. Strumson do next?"]
    assert drop_trailing_question(text) == "The rocket lifts off."
    assert drop_trailing_question("What does Dr. Fox see?") == ""


def test_streaming_does_not_break_after_mrs():
    acc = SentenceAccumulator()
    out = []
    for piece in ["The rocket lifts off. What does Mrs", ". Strumson do ", "next?"]:
        out += acc.feed(piece)
    out.append(acc.flush())
    assert out == ["The rocket lifts off.", "What does Mrs. Strumson do next?"]


def test_ordinary_sentences_still_split():
    assert split_sentences("Pip ran. The fox slept! Where did they go?") == ["Pip ran.", "The fox slept!", "Where did they go?"]
    assert split_sentences("It costs 3.5 coins. Wow.") == ["It costs 3.5 coins.", "Wow."]


def test_sound_tags_are_stripped():
    from yarnotron.services.transcript_clean import strip_sound_tags

    assert strip_sound_tags("*crickets*\n Can we continue? [BLANK_AUDIO]") == "Can we continue?"
    assert strip_sound_tags("*thud*") == ""
    assert strip_sound_tags("The fox (quietly) ran.") == "The fox ran."


def test_timing_record_goes_to_the_data_volume(tmp_path, monkeypatch):
    import json

    from yarnotron.core.timing import LatencyTimer

    monkeypatch.setenv("YARNOTRON_DATA_DIR", str(tmp_path))
    timer = LatencyTimer("agent_turn", session_id="s1")
    timer.log()
    row = json.loads((tmp_path / "timing.jsonl").read_text().splitlines()[0])
    assert row["session_id"] == "s1" and "at" in row
