from yarnotron.agent.name_match import resolve_names, similarity
from yarnotron.models.story import Character, Scene, Story

STORY = Story(title="The Super Solar System",
              characters=[Character(id="c1", name="Mrs. Strumson", role="explorer"), Character(id="c2", name="Valentina")],
              scenes=[Scene(id="s1", title="Finding Super Mercury", summary="")])


def test_garbled_surnames_resolve_to_the_character():
    for heard in ("Drumston", "Shumson", "Stratson", "Strumston"):
        text, matches = resolve_names(f"Can we not talk about Mrs. {heard} anymore", STORY)
        assert "Mrs. Strumson" in text and heard not in text, (heard, text)
        assert [m.item_id for m in matches] == ["c1"]


def test_correct_names_and_unrelated_words_are_left_alone():
    text, matches = resolve_names("Mrs. Strumson lands on the planet", STORY)
    assert text == "Mrs. Strumson lands on the planet" and [m.item_id for m in matches] == ["c1"], "said right: reported, not rewritten"
    text, matches = resolve_names("The rocket flew past a big yellow star", STORY)
    assert text == "The rocket flew past a big yellow star" and matches == []


def test_loose_matching_needs_a_removal_style_turn_and_a_matching_honorific():
    text, matches = resolve_names("Can you just not talk about Mrs. Johnston?", STORY)
    assert matches == [], "on an ordinary turn a different surname is a different person"
    text, matches = resolve_names("Can you just not talk about Mrs. Johnston?", STORY, loose=True)
    assert "Mrs. Strumson" in text and [m.item_id for m in matches] == ["c1"]
    text, matches = resolve_names("Remove Mr. Johnston", STORY, loose=True)
    assert matches == [], "Mr. is not Mrs."


def test_a_garbled_first_name_matches_too():
    text, matches = resolve_names("Valentine is the president", STORY)
    assert text == "Valentina is the president" and matches[0].item_id == "c2"


def test_similarity_orders_sensibly():
    assert similarity("drumston", "strumson") > 0.74
    assert similarity("banana", "strumson") < 0.5


def test_removal_intent_fires_on_requests_not_on_narration():
    from yarnotron.agent.agent_engine import _REMOVAL_INTENT as R

    for yes in ("Delete Valentina", "Can we delete Mrs. Strumson", "Can you just not talk about Mrs. Johnston?",
                "I don't want her in the story!", "Mrs. Drumster is not mixed for she is deleted!",
                "Can we not talk about Mrs. Drumston and just not have her", "First of all, can we remove Mr.",
                "please take her out", "get rid of the dragon"):
        assert R.search(yes), yes
    for no in ("The knight removed his helmet", "She did not want to go home", "There was no more food",
               "The fog is gone", "He had to remove the stone", "They did not have any gold"):
        assert not R.search(no), no
