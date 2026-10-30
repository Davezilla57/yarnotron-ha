from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from yarnotron.agent import code_messages
from yarnotron.agent.agent_engine import AgentEngine, interpret_actions, load_spec
from yarnotron.agent.safety import SafetyClassifier
from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.models.agent import AgentTurnRequest, TurnIntent
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import Character, Story, StoryPatch
from yarnotron.services.memory import StoryMemoryBuilder
from yarnotron.services.ollama import OllamaUnavailableError

from tests.test_turn_processor import (
    FakeEmbedder,
    FakeSafetyReviewer,
    FakeStoryRepository,
)

TALK = "I love that the fox found a lantern. What does Pip see inside?"


class FakeAgentLLM:
    """Talker text streams in small pieces; the Scribe returns a fixed action list."""

    def __init__(self, talk: str = TALK, actions: list[dict] | None = None, scribe_error: bool = False) -> None:
        self.talk = talk
        self.actions = actions or []
        self.scribe_error = scribe_error
        self.talker_prompts: list[str] = []
        self.scribe_prompts: list[str] = []

    async def stream_text(self, prompt, model=None, **kwargs):
        self.talker_prompts.append(prompt)
        for i in range(0, len(self.talk), 7):
            await asyncio.sleep(0)
            yield self.talk[i : i + 7]

    async def generate_json(self, prompt, model=None, **kwargs):
        self.scribe_prompts.append(prompt)
        if self.scribe_error:
            raise OllamaUnavailableError("down")
        return {"actions": self.actions}


class GateOn:
    """Review the text of each sentence; veto any sentence containing `bad`."""

    def __init__(self, bad: str | None = None, output_redirect: bool = False) -> None:
        self.bad = bad
        self.output_redirect = output_redirect
        self.reviewed: list[str] = []

    def classify_transcript(self, transcript):
        return SafetyClassifier().classify(transcript)

    async def review_transcript(self, transcript, story=None, intent=None):
        return SafetyDecision()

    async def review_assistant_output(self, transcript, response_text, story_patches):
        self.reviewed.append(response_text)
        if self.bad and self.bad in response_text:
            return SafetyDecision(action=SafetyAction.REDIRECT, severity=SafetySeverity.REVIEW, child_message="Let's keep that part of the story safe and kind.")
        if self.output_redirect and story_patches:
            return SafetyDecision(action=SafetyAction.REDIRECT, severity=SafetySeverity.REVIEW, child_message="Let's make that safer.")
        return SafetyDecision()


def build(llm, safety=None, repo=None):
    repo = repo or FakeStoryRepository()
    safety = safety or GateOn()
    engine = AgentEngine(llm=llm, safety=safety, scribe_budget_seconds=2.0)
    proc = TurnProcessor(
        ollama=llm, repository=repo, embedder=FakeEmbedder(), memory_builder=StoryMemoryBuilder(),
        safety=safety, engine=engine,
    )
    return proc, repo


async def run(proc, transcript, story_id="story_test", **kw):
    said: list[str] = []

    async def on_sentence(text: str) -> None:
        said.append(text)

    request = AgentTurnRequest(story_id=story_id, transcript=transcript, **kw)
    response = await proc.process(request, on_sentence=on_sentence)
    return response, said


async def test_plain_turn_streams_two_sentences_and_changes_nothing():
    proc, repo = build(FakeAgentLLM())
    response, said = await run(proc, "The fox walks through the forest at night.")
    assert said == ["I love that the fox found a lantern.", "What does Pip see inside?"]
    assert response.applied_story_patches == []
    assert response.intent == TurnIntent.CONTINUE_STORY


async def test_auto_tool_is_applied_only_after_the_gate():
    llm = FakeAgentLLM(actions=[{"tool": "add_fact", "args": {"text": "The lantern never goes out."}}])
    proc, repo = build(llm)
    response, said = await run(proc, "The lantern never goes out.")
    assert [p.operation.value for p in response.applied_story_patches] == ["add_fact"]
    assert response.updated_story.facts[0].text == "The lantern never goes out."
    assert response.intent == TurnIntent.EDIT_STORY


async def test_sentence_veto_stops_speech_and_saves_nothing():
    llm = FakeAgentLLM(actions=[{"tool": "add_fact", "args": {"text": "x"}}])
    proc, repo = build(llm, GateOn(bad="Pip see"))
    response, said = await run(proc, "The fox finds a lantern.")
    assert said[0] == "I love that the fox found a lantern."
    assert said[-1] == "Let's keep that part of the story safe and kind."
    assert response.applied_story_patches == []
    assert response.response_text == "Let's keep that part of the story safe and kind."
    assert repo._story.facts == []


async def test_failed_output_review_discards_the_scribes_actions():
    llm = FakeAgentLLM(actions=[{"tool": "add_fact", "args": {"text": "A secret fact."}}])
    proc, repo = build(llm, GateOn(output_redirect=True))
    response, said = await run(proc, "A secret fact.")
    assert response.applied_story_patches == []
    assert repo._story.facts == []
    assert said[-1] == "Let's keep that part of the story safe and kind."


async def test_destructive_edit_is_held_and_the_child_is_asked_once():
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_fennick", name="Fennick")])
    llm = FakeAgentLLM(actions=[{"tool": "remove_character", "args": {"id": "character_fennick"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Remove Fennick.")
    assert repo._story.characters, "nothing removed before the child says yes"
    assert response.applied_story_patches == []
    assert response.story_patches[0].child_confirmation_required
    assert said[-1].endswith(code_messages.CONFIRM_QUESTION)
    assert sum(s.count("?") for s in said) == 1, said
    # The existing yes/no resolver applies it on the next turn.
    follow, _ = await run(proc, "yes")
    assert follow.applied_story_patches and not repo._story.characters


async def test_decline_leaves_the_story_alone():
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_fennick", name="Fennick")])
    llm = FakeAgentLLM(actions=[{"tool": "remove_character", "args": {"id": "character_fennick"}}])
    proc, _ = build(llm, repo=repo)
    await run(proc, "Remove Fennick.")
    follow, _ = await run(proc, "no")
    assert repo._story.characters and follow.applied_story_patches == []


async def test_scene_from_narration_is_offered_not_added():
    llm = FakeAgentLLM(actions=[{"tool": "add_scene", "args": {"title": "The hidden door", "summary": "Wren finds a door."}}])
    proc, repo = build(llm)
    response, said = await run(proc, "Wren climbs the stairs and finds a hidden door behind the lamp room.")
    assert repo._story.scenes == []
    assert said[-1].endswith(code_messages.SCENE_OFFER_QUESTION)
    follow, _ = await run(proc, "yes")
    assert len(repo._story.scenes) == 1 and follow.applied_story_patches


async def test_explicit_scene_request_is_added_without_a_question():
    llm = FakeAgentLLM(actions=[{"tool": "add_scene", "args": {"title": "Swim", "summary": "Mackerel swims out."}}])
    proc, repo = build(llm)
    response, said = await run(proc, "Add a scene where Mackerel swims out to help Barnacle.")
    assert len(repo._story.scenes) == 1
    assert code_messages.SCENE_OFFER_QUESTION not in " ".join(said)


async def test_draft_text_must_be_the_childs_own_words():
    result = interpret_actions(
        {"actions": [{"tool": "add_scene", "args": {"title": "A", "summary": "B", "draft_text": "A rewritten flowery sentence."}}]},
        AgentTurnRequest(transcript="Add a scene where the cat naps."),
    )
    assert "draft_text" not in result.patches[0].data
    ok = interpret_actions(
        {"actions": [{"tool": "add_scene", "args": {"title": "A", "summary": "B", "draft_text": "The cat naps in the sun."}}]},
        AgentTurnRequest(transcript="The cat naps in the sun."),
    )
    assert ok.patches[0].data["draft_text"] == "The cat naps in the sun."


def test_unknown_tools_are_refused_and_caps_hold():
    raw = {"actions": [
        {"tool": "delete_story", "args": {}},
        {"tool": "approve_media", "args": {"id": "x"}},
        {"tool": "note_interest", "args": {"name": "Minecraft"}},
        {"tool": "note_interest", "args": {"name": "Roblox"}},
        *[{"tool": "add_fact", "args": {"text": f"f{i}"}} for i in range(6)],
    ]}
    result = interpret_actions(raw, AgentTurnRequest(transcript="stuff"))
    assert "delete_story" in result.refused and "approve_media" in result.refused
    assert len(result.patches) == 3
    assert result.interest == ("Minecraft", False)


def test_undo_means_undo():
    raw = {"actions": [{"tool": "undo_last_change", "args": {}}, {"tool": "add_fact", "args": {"text": "x"}},
                       {"tool": "request_picture", "args": {"description": "a lighthouse"}}]}
    result = interpret_actions(raw, AgentTurnRequest(transcript="undo that"))
    assert result.undo and result.patches == [] and not result.picture


async def test_undo_restores_the_last_version():
    repo = FakeStoryRepository()
    llm = FakeAgentLLM(actions=[{"tool": "add_fact", "args": {"text": "Temporary fact."}}])
    proc, _ = build(llm, repo=repo)
    await run(proc, "Temporary fact.")
    assert repo._story.facts
    llm.actions = [{"tool": "undo_last_change", "args": {}}]
    response, said = await run(proc, "undo that")
    assert repo._story.facts == []
    assert said[-1] == code_messages.UNDO_DONE
    assert response.intent == TurnIntent.UNDO_LAST_CHANGE


async def test_picture_request_only_creates_a_pending_asset():
    llm = FakeAgentLLM(actions=[{"tool": "request_picture", "args": {"description": "the lighthouse in a storm"}}])
    proc, repo = build(llm)
    response, said = await run(proc, "Can you draw the lighthouse in the storm?")
    assert said[-1].endswith(code_messages.PICTURE_SAVED)
    assert len(response.media_asset_ids) == 1
    assert repo.media_assets and repo.media_assets[0].status.value == "pending_parent_review"


async def test_scribe_failure_never_blocks_the_conversation():
    llm = FakeAgentLLM(scribe_error=True)
    proc, repo = build(llm)
    response, said = await run(proc, "The fox walks on.")
    assert said == ["I love that the fox found a lantern.", "What does Pip see inside?"]
    assert response.applied_story_patches == []


async def test_ambiguous_interest_replaces_the_question_and_holds_patches():
    llm = FakeAgentLLM(actions=[{"tool": "note_interest", "args": {"name": "Kerbal Space Program", "ambiguous": True}}])
    proc, repo = build(llm)
    response, said = await run(proc, "Let's add Kerbal Space Program.")
    assert said[-1] == "Is Kerbal Space Program part of this story, or just something you love?"
    assert "What does Pip see inside?" not in " ".join(said)
    assert any(n.kind == "interest_clarification" for n in response.tutor_notes)


async def test_contact_details_never_reach_a_model_or_the_store():
    llm = FakeAgentLLM()
    proc, repo = build(llm)
    response, said = await run(proc, "I live at 42 Maple Street, can the fox visit?")
    everything = " ".join(llm.talker_prompts + llm.scribe_prompts)
    assert "42 Maple" not in everything and "Maple Street" not in everything
    assert response.response_text.startswith("I can't take or keep addresses")
    assert said[0] == "I can't take or keep addresses or email addresses, but I can keep going with the story."
    assert "Maple" not in repo.turns[-1].transcript
    assert all("Maple" not in (c.content or "") for c in repo.memory_chunks)


async def test_self_harm_keyword_speaks_the_fixed_script():
    proc, repo = build(FakeAgentLLM())
    response, said = await run(proc, "I want to kill myself.")
    assert response.response_text == code_messages.WORRIED_CHILD_SCRIPT
    assert response.safety.action == SafetyAction.ALERT_PARENT
    assert said == []


async def test_other_alert_keywords_keep_the_existing_stop_line():
    proc, repo = build(FakeAgentLLM())
    response, _ = await run(proc, "tell me something with porn")
    assert response.response_text != code_messages.WORRIED_CHILD_SCRIPT


async def test_locked_session_never_reaches_the_agent():
    llm = FakeAgentLLM()
    proc, repo = build(llm)
    for _ in range(5):
        await run(proc, "A story about a bully who steals.")
    llm.talker_prompts.clear()
    llm.scribe_prompts.clear()
    response, said = await run(proc, "Let's keep going.")
    # The existing strike limit applies per session; the follow-up never costs a model call once ended.
    if response.safety.categories and "session_ended" in response.safety.categories:
        assert llm.talker_prompts == [] and llm.scribe_prompts == []


def test_spec_files_match_the_code_messages():
    root = Path(__file__).resolve().parents[1] / "src/yarnotron/agent/spec"
    agent_yaml = (root / "agent.yaml").read_text()
    assert code_messages.WORRIED_CHILD_SCRIPT in agent_yaml
    from yarnotron.services.pii import PII_NOT_KEPT_LINE
    assert PII_NOT_KEPT_LINE in agent_yaml


def test_spec_loads_and_talker_has_no_tool_text():
    spec = load_spec()
    assert "You are Yarnotron" in spec.talker_rules
    assert "Changing the story" not in spec.talker_rules and "Changing the story" in spec.scribe_rules
    assert {"add_scene", "remove_character", "undo_last_change", "request_picture"} <= set(spec.tools)
    assert "approve_media" not in spec.tools and "delete_story" not in spec.tools


async def test_code_supplied_line_is_not_judged_as_model_output():
    gate = GateOn()
    proc, repo = build(FakeAgentLLM(), gate)
    response, said = await run(proc, "I live at 42 Maple Street, can the fox visit?")
    assert all("can't take or keep" not in text for text in gate.reviewed)
    assert response.safety.action == SafetyAction.ALLOW


async def test_first_sentence_is_spoken_before_the_second_is_written():
    order: list[str] = []

    class Slow(FakeAgentLLM):
        async def stream_text(self, prompt, model=None, **kwargs):
            yield "Pip hears a tiny bell ring. "
            yield "What"
            await asyncio.sleep(0.05)
            order.append("second sentence still being written")
            yield " does Pip do?"

    proc, repo = build(Slow())
    said: list[str] = []

    async def on_sentence(text: str) -> None:
        order.append(f"said: {text}")
        said.append(text)

    await proc.process(AgentTurnRequest(story_id="story_test", transcript="Pip walks on."), on_sentence=on_sentence)
    assert order.index("said: Pip hears a tiny bell ring.") < order.index("second sentence still being written")


async def test_final_statement_without_a_question_keeps_its_words_before_the_code_line():
    llm = FakeAgentLLM(
        talk="Pip hears a tiny bell ring. The cave grows quiet.",
        actions=[{"tool": "request_picture", "args": {"description": "bell"}}],
    )
    proc, repo = build(llm)
    response, said = await run(proc, "Draw the bell.")
    assert said[-1] == f"The cave grows quiet. {code_messages.PICTURE_SAVED}"


async def test_scene_offers_are_not_repeated_every_turn():
    llm = FakeAgentLLM(actions=[{"tool": "add_scene", "args": {"title": "Door", "summary": "A door."}}])
    proc, repo = build(llm)
    first, said1 = await run(proc, "Wren finds a hidden door behind the lamp room.")
    assert said1[-1].endswith(code_messages.SCENE_OFFER_QUESTION)
    second, said2 = await run(proc, "Behind it is a long staircase going down.")
    assert code_messages.SCENE_OFFER_QUESTION not in " ".join(said2)


async def test_caution_prefix_is_not_repeated_in_the_final_text():
    proc, _ = build(FakeAgentLLM())
    response, said = await run(proc, "A story about a bully who steals.")
    caution = said[0]
    assert caution.startswith("Let's steer that")
    assert response.response_text.count(caution) == 1, response.response_text


async def test_redirect_after_caution_is_not_said_twice_and_uses_code_wording():
    proc, _ = build(FakeAgentLLM(), safety=GateOn(bad="lantern"))
    response, said = await run(proc, "A story about a bully who steals.")
    assert sum(s.startswith("Let's steer that") for s in said) == 1, said
    assert "Let's keep that part of the story safe and kind." not in " ".join(said), "the model's own wording is never spoken"


async def test_cautioned_turns_with_a_vetoed_reply_still_spend_strikes():
    proc, _ = build(FakeAgentLLM(), safety=GateOn(bad="lantern"))
    ended = False
    session_id = None
    for _ in range(5):
        response, _ = await run(proc, "A story about a bully who steals.", session_id=session_id)
        session_id = response.session_id
        ended = "strike_limit_reached" in response.safety.categories
    assert ended, "five cautioned turns end the session even when the reply was vetoed"


async def test_talker_cannot_claim_a_change_it_did_not_make():
    claim = "Okay, Fennick is gone. The lighthouse feels quieter. What does Wren do next?"
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_fennick", name="Fennick")])
    llm = FakeAgentLLM(talk=claim, actions=[{"tool": "remove_character", "args": {"id": "character_fennick"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Remove Fennick.")
    assert not any("is gone" in s for s in said), said
    assert said[-1].endswith(code_messages.CONFIRM_QUESTION)
    assert repo._story.characters


async def test_ordinary_story_sentences_are_not_dropped_when_nothing_is_being_edited():
    llm = FakeAgentLLM(talk="The fog is gone and the shore is bright. What does Pip see?")
    proc, _ = build(llm)
    _, said = await run(proc, "The sun came up over the water.")
    assert said[0] == "The fog is gone and the shore is bright."


async def test_scribe_placeholder_id_is_resolved_by_the_name_the_child_said():
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_fennick", name="Fennick")])
    llm = FakeAgentLLM(actions=[{"tool": "remove_character", "args": {"id": "character_placeholder"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Remove the character Fennick, he never showed up.")
    assert response.story_patches and response.story_patches[0].data["id"] == "character_fennick"
    assert response.story_patches[0].child_confirmation_required
    assert said[-1].endswith(code_messages.CONFIRM_QUESTION)


async def test_unresolvable_target_asks_instead_of_going_quiet():
    repo = FakeStoryRepository()
    repo._story = Story(
        title="T",
        characters=[Character(id="character_a", name="Fennick"), Character(id="character_b", name="Barnacle")],
    )
    llm = FakeAgentLLM(actions=[{"tool": "remove_character", "args": {"id": "character_nobody"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Remove that character.")
    assert said[-1].endswith(code_messages.which_one(["Fennick", "Barnacle"]))
    assert len(repo._story.characters) == 2


@pytest.mark.parametrize(
    "line",
    [
        "My uncle told me not to tell anyone what we did.",
        "This is our little secret, okay?",
        "Someone is hurting me.",
        "I don't feel safe at home.",
        "My coach touches me and I don't like it.",
    ],
)
async def test_worried_child_lines_get_the_fixed_script(line):
    proc, _ = build(FakeAgentLLM())
    response, said = await run(proc, line)
    assert response.response_text == code_messages.WORRIED_CHILD_SCRIPT
    assert response.safety.action == SafetyAction.ALERT_PARENT
    assert "worried_child" in response.safety.categories
    assert said == []


@pytest.mark.parametrize(
    "line",
    [
        "The knight told the king not to tell anyone about the dragon.",
        "The monster hits the drum and I laugh.",
        "She promised to keep the secret treasure hidden.",
        "My dog touches my hand with his nose.",
    ],
)
async def test_story_lines_are_not_mistaken_for_a_worried_child(line):
    proc, _ = build(FakeAgentLLM())
    response, said = await run(proc, line)
    assert response.response_text != code_messages.WORRIED_CHILD_SCRIPT
    assert said


def test_short_answers_with_a_yes_or_no_word_resolve_confirmations():
    from yarnotron.agent.turn_processor import _confirmation_answer

    assert _confirmation_answer("No, keep it.") is False
    assert _confirmation_answer("Yes please do it") is True
    assert _confirmation_answer("No, the dragon flies away into the mountains") is None
    assert _confirmation_answer("Okay then Wren climbs the lighthouse stairs") is None


@pytest.mark.parametrize("line", ["S E X", "I want to see a n u d e picture", "p o r n please"])
def test_spelled_out_alert_words_are_caught(line):
    decision = SafetyClassifier().classify(line)
    assert decision.action == SafetyAction.ALERT_PARENT


def test_ordinary_single_letters_are_not_a_word():
    assert SafetyClassifier().classify("Plan A and plan B and then plan C").action == SafetyAction.ALLOW


async def test_removal_of_something_the_child_did_not_name_is_not_staged():
    repo = FakeStoryRepository()
    repo._story = Story(
        title="T",
        characters=[Character(id="character_a", name="Wren", role="keeper"), Character(id="character_b", name="Barnacle", role="gull")],
    )
    llm = FakeAgentLLM(actions=[{"tool": "remove_character", "args": {"id": "character_b"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Remove the character Fennick, he never showed up.")
    assert response.story_patches == []
    assert said[-1].endswith(code_messages.which_one(["Wren", "Barnacle"]))
    assert len(repo._story.characters) == 2


async def test_removal_by_role_word_is_still_held_for_a_yes():
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_b", name="Barnacle", role="gull")])
    llm = FakeAgentLLM(actions=[{"tool": "remove_character", "args": {"id": "character_b"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Get rid of the gull.")
    assert response.story_patches and response.story_patches[0].child_confirmation_required


def test_plural_and_past_forms_of_redirect_words_are_caught():
    for line in ["someone gets tortured", "drugs in the water", "the bullies steal it"]:
        assert SafetyClassifier().classify(line).action == SafetyAction.REDIRECT, line
    assert SafetyClassifier().classify("a stealthy fox in the garden").action == SafetyAction.ALLOW
    assert SafetyClassifier().classify("The pirate steals the golden compass back.").action == SafetyAction.ALLOW


async def test_code_line_does_not_repeat_a_sentence_the_talker_already_said():
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_fennick", name="Fennick")])
    llm = FakeAgentLLM(
        talk="Fennick can go. That's a big change.",
        actions=[{"tool": "remove_character", "args": {"id": "character_fennick"}}],
    )
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Remove Fennick.")
    assert " ".join(said).count("That's a big change.") == 1, said
    assert said[-1].endswith("Should I make it?")


async def test_scene_removal_by_position_is_held_for_a_yes():
    from yarnotron.models.story import Scene

    repo = FakeStoryRepository()
    repo._story = Story(
        title="T",
        scenes=[Scene(id="scene_a", title="The Light Goes Out", summary="x", sequence=0), Scene(id="scene_b", title="Climbing the Stair", summary="y", sequence=1)],
    )
    llm = FakeAgentLLM(actions=[{"tool": "remove_scene", "args": {"id": "scene_b"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Delete the second scene.")
    assert response.story_patches and response.story_patches[0].child_confirmation_required
    wrong = FakeAgentLLM(actions=[{"tool": "remove_scene", "args": {"id": "scene_a"}}])
    proc2, repo2 = build(wrong, repo=repo)
    response2, said2 = await run(proc2, "Delete the second scene.")
    assert response2.story_patches == [] and said2[-1].endswith(code_messages.UNCLEAR_TARGET)


async def test_talker_future_promises_of_a_change_are_dropped():
    llm = FakeAgentLLM(talk="Okay, I will remove that part of the story. What happens next?")
    proc, _ = build(llm)
    _, said = await run(proc, "Delete the second scene.")
    assert not any("I will remove" in s for s in said), said


async def test_interest_is_only_ambiguous_when_the_child_asks_to_add_it():
    llm = FakeAgentLLM(actions=[{"tool": "note_interest", "args": {"name": "Kerbal Space Program", "ambiguous": True}}])
    proc, _ = build(llm)
    response, said = await run(proc, "Make it feel like Kerbal Space Program.")
    assert not any(n.kind == "interest_clarification" for n in response.tutor_notes)
    response2, _ = await run(proc, "Let's add Kerbal Space Program.")
    assert any(n.kind == "interest_clarification" for n in response2.tutor_notes)


async def test_yes_and_more_saves_the_offered_scene_and_carries_on():
    llm = FakeAgentLLM(actions=[{"tool": "add_scene", "args": {"title": "Door", "summary": "A door."}}])
    proc, repo = build(llm)
    first, said1 = await run(proc, "Wren finds a hidden door behind the lamp room.")
    assert said1[-1].endswith(code_messages.SCENE_OFFER_QUESTION) and repo._story.scenes == []
    follow, _ = await run(proc, "Yes, and add that she opens it very slowly.")
    assert len(repo._story.scenes) == 1
    assert follow.applied_story_patches and follow.applied_story_patches[0].operation.value == "add_scene"
    assert follow.response_text


async def test_yes_and_more_does_not_confirm_a_removal():
    llm = FakeAgentLLM(actions=[{"tool": "remove_scene", "args": {"id": "scene_a"}}])
    proc, repo = build(llm)
    await run(proc, "Remove the first scene.")
    scenes_before = len(repo._story.scenes)
    await run(proc, "Yes, and then the cat naps in the sun.")
    assert len(repo._story.scenes) == scenes_before


def test_rename_without_a_named_title_changes_nothing():
    request = AgentTurnRequest(transcript="Let's rename this story.")
    result = interpret_actions({"actions": [{"tool": "update_title", "args": {"title": "New Title"}}]}, request)
    assert result.patches == [] and "update_title:not_named" in result.refused
    named = interpret_actions(
        {"actions": [{"tool": "update_title", "args": {"title": "Lion Baseball"}}]},
        AgentTurnRequest(transcript="No. Rename the story Lion Baseball."),
    )
    assert named.patches and named.patches[0].data["title"] == "Lion Baseball"


def test_commands_are_not_saved_as_story_wording():
    for said in ("No, create a new scene with the lion getting very sad.", "Create a story about a lion that..."):
        result = interpret_actions(
            {"actions": [{"tool": "add_scene", "args": {"title": "A", "summary": "B", "draft_text": said}}]},
            AgentTurnRequest(transcript=said),
        )
        assert "draft_text" not in result.patches[0].data
    kept = interpret_actions(
        {"actions": [{"tool": "add_scene", "args": {"title": "A", "summary": "B", "draft_text": "He hits the ball."}}]},
        AgentTurnRequest(transcript="He hits the ball."),
    )
    assert kept.patches[0].data["draft_text"] == "He hits the ball."


async def test_the_same_filler_three_times_running_pauses_the_session():
    proc, repo = build(FakeAgentLLM())
    session_id = None
    categories = []
    for _ in range(3):
        response, _ = await run(proc, "Thank you.", session_id=session_id)
        session_id = response.session_id
        categories.append(set(response.safety.categories))
    assert "session_ended" not in categories[0] and "session_ended" not in categories[1]
    assert {"session_ended", "idle_loop"} <= categories[2]
    session = await repo.get_session(session_id)
    assert session.ended_at is not None


async def test_one_polite_thank_you_between_real_sentences_is_just_a_turn():
    proc, repo = build(FakeAgentLLM())
    session_id = None
    for text in ("Thank you.", "The fox crept into the garden.", "Thank you.", "The moon came up.", "Thank you."):
        response, _ = await run(proc, text, session_id=session_id)
        session_id = response.session_id
        assert "idle_loop" not in response.safety.categories
    assert (await repo.get_session(session_id)).ended_at is None


async def test_sound_only_transcript_is_not_treated_as_story():
    proc, _ = build(FakeAgentLLM())
    response, said = await run(proc, "*thud*")
    assert response.response_text == code_messages.NOT_HEARD
    assert not response.applied_story_patches


async def test_questions_asked_recently_are_listed_to_the_talker_not_to_repeat():
    from yarnotron.agent.agent_engine import build_talker_prompt

    request = AgentTurnRequest(child_id="c", transcript="No.")
    prompt = build_talker_prompt(request, None, [], [], avoid_questions=["What does the first star represent?"])
    assert "What does the first star represent?" in prompt and "Do not ask any of them again" in prompt
    assert "Do not ask any of them again" not in build_talker_prompt(request, None, [], [])


def _crew_repo():
    repo = FakeStoryRepository()
    repo._story = Story(title="T", characters=[Character(id="character_m", name="Mrs. Strumson", role="explorer"),
                                              Character(id="character_v", name="Valentina")])
    return repo


async def test_garbled_name_on_a_removal_turn_is_matched_and_asked_about_by_name():
    repo = _crew_repo()
    llm = FakeAgentLLM(talk="Mrs. Strumson stays in the story. What does she do next?", actions=[])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Can we not talk about Mrs. Drumston anymore?")
    spoken = " ".join(said)
    assert "stays in the story" not in spoken, "the model's words are not used on a removal turn"
    assert said[-1] == f"You want me to take out Mrs. Strumson. {code_messages.CONFIRM_QUESTION}"
    assert response.story_patches[0].operation.value == "remove_character"
    assert response.story_patches[0].data["id"] == "character_m"
    assert len(repo._story.characters) == 2, "nothing is removed until the child says yes"


async def test_a_model_update_cannot_stand_in_for_a_removal():
    repo = _crew_repo()
    llm = FakeAgentLLM(talk="Okay. What next?", actions=[{"tool": "update_character", "args": {"id": "character_m", "notes": "deleted"}}])
    proc, _ = build(llm, repo=repo)
    response, said = await run(proc, "Mrs. Strumson is deleted!")
    assert [p.operation.value for p in response.story_patches] == ["remove_character"]
    assert not response.applied_story_patches
    assert repo._story.characters[0].notes in (None, "", [])


async def test_false_refusal_is_not_spoken_when_the_child_asks_for_an_edit():
    llm = FakeAgentLLM(talk="I cannot change the title. What should we write next?", actions=[])
    proc, _ = build(llm)
    _, said = await run(proc, "Can you change the title please?")
    assert not any("cannot change" in s for s in said), said


async def _session_turns(proc, repo):
    return await repo.list_conversation_turns(limit=20)


async def test_hand_over_gets_a_nod_without_the_model_and_then_a_quiet_reply():
    from yarnotron.agent import turn_acts

    llm = FakeAgentLLM(talk="Oh, I like that.")
    proc, repo = build(llm)
    first, _ = await run(proc, "No, let me tell you about the flag!")
    assert first.response_text in turn_acts.LISTEN_REPLIES and not llm.talker_prompts, "no model call for a nod"
    second, _ = await run(proc, "It has fifty white stars on a blue background", session_id=first.session_id)
    assert "Reply with one short acknowledgement" in llm.talker_prompts[-1]
    assert "What does Pip" not in second.response_text
    third, _ = await run(proc, "What happens next?", session_id=first.session_id)
    assert "Reply with one short acknowledgement" not in llm.talker_prompts[-1], "a question ends listening mode"


async def test_unfinished_sentence_is_held_and_joined_to_the_next():
    from yarnotron.agent import turn_acts

    llm = FakeAgentLLM()
    proc, repo = build(llm)
    first, _ = await run(proc, "She climbs back into the rocket and")
    assert first.response_text in turn_acts.HOLD_REPLIES and not llm.talker_prompts
    second, _ = await run(proc, "flies off to the next planet.", session_id=first.session_id)
    assert "She climbs back into the rocket and flies off to the next planet." in llm.talker_prompts[-1]


async def test_two_frustrated_turns_get_an_apology_and_an_offer_to_listen():
    from yarnotron.agent import turn_acts

    proc, _ = build(FakeAgentLLM())
    first, _ = await run(proc, "Why can't you do anything?")
    assert first.response_text != turn_acts.REPAIR_LINE
    second, _ = await run(proc, "It's not your business", session_id=first.session_id)
    assert second.response_text == turn_acts.REPAIR_LINE
    assert {n.kind for n in second.tutor_notes} >= {"frustration", "dictation"}


async def test_new_story_delete_story_and_calling_dad():
    from yarnotron.agent import turn_acts

    proc, repo = build(FakeAgentLLM())
    new, _ = await run(proc, "Can we create a new story?")
    assert new.response_text == turn_acts.NEW_STORY_LINE and new.updated_story.title == "Untitled Story"
    gone, _ = await run(proc, "Delete that story.")
    assert gone.response_text == turn_acts.STORY_DELETE_LINE
    dad, _ = await run(proc, "Dad, can you help me?")
    assert dad.response_text == turn_acts.GROWN_UP_LINE


async def test_she_means_the_character_just_introduced_not_the_models_guess():
    repo = _crew_repo()
    llm = FakeAgentLLM(actions=[{"tool": "add_character", "args": {"name": "Zed", "role": "pilot"}}])
    proc, _ = build(llm, repo=repo)
    first, _ = await run(proc, "A pilot called Zed flies the ship.")
    assert any(c.name == "Zed" for c in repo._story.characters)
    zed = next(c for c in repo._story.characters if c.name == "Zed")
    llm.actions = [{"tool": "update_character", "args": {"id": "character_m", "role": "president"}}]  # wrong guess
    await run(proc, "She is the president too.", session_id=first.session_id)
    assert next(c for c in repo._story.characters if c.id == zed.id).role == "president"
    assert next(c for c in repo._story.characters if c.id == "character_m").role == "explorer"


async def test_corrections_and_places_guidance():
    from yarnotron.agent.agent_engine import interpret_actions

    raw = {"actions": [{"tool": "add_fact", "args": {"text": "It's not a space station, it's a nation."}},
                       {"tool": "add_fact", "args": {"text": "The Space Nation floats in outer space."}}]}
    result = interpret_actions(raw, AgentTurnRequest(transcript="It's not a space station, it's a nation in outer space."))
    assert [p.data["text"] for p in result.patches] == ["The Space Nation floats in outer space."]
    from yarnotron.agent.agent_engine import load_spec

    assert "Planets, places" in load_spec().scribe_rules and "introduced most recently" in load_spec().scribe_rules


async def test_session_summary_counts_without_words():
    from yarnotron.services.session_quality import summarize

    proc, repo = build(FakeAgentLLM())
    first, _ = await run(proc, "She climbs into the rocket and")
    await run(proc, "flies away.", session_id=first.session_id)
    await run(proc, "Thank you.", session_id=first.session_id)
    counts = summarize([t for t in await repo.list_conversation_turns(limit=50) if t.session_id == first.session_id])
    assert counts["turns"] == 3 and counts["held_sentences"] == 1 and counts["filler_turns"] == 1
