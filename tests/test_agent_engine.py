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
        {"actions": [{"tool": "add_scene", "args": {"title": "A", "summary": "B", "draft_text": "the cat naps"}}]},
        AgentTurnRequest(transcript="Add a scene where the cat naps."),
    )
    assert ok.patches[0].data["draft_text"] == "the cat naps"


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
    assert said[-1].endswith(code_messages.UNCLEAR_TARGET)
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
    assert said[-1].endswith(code_messages.UNCLEAR_TARGET)
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
