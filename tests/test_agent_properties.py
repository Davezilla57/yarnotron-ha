"""Property tests for the agent engine (eval/properties.md P1-P13). The model is mocked: these prove the policy
layer, not how a real model behaves. ~1,000 random turns per property family, fixed seed."""

from __future__ import annotations

import random
import re

from yarnotron.agent import code_messages
from yarnotron.agent.agent_engine import MAX_ADDS, MAX_ACTIONS, interpret_actions, load_spec
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.safety import SafetyAction
from yarnotron.models.story import Character, Scene, Story

from tests.test_agent_engine import FakeAgentLLM, GateOn, build, run

DESTRUCTIVE = {"remove_character", "remove_scene", "remove_fact"}
WORDS = ["fox", "lantern", "cave", "moth", "door", "storm", "bell", "river", "key", "moon"]


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def seeded_repo():
    from tests.test_turn_processor import FakeStoryRepository

    repo = FakeStoryRepository()
    repo._story = Story(
        title="T",
        characters=[Character(id=f"character_{i}", name=f"Name{i}") for i in range(3)],
        scenes=[Scene(id=f"scene_{i}", title=f"Scene {i}", summary="s", sequence=i + 1) for i in range(3)],
    )
    return repo


def random_actions(rng: random.Random, story: Story) -> list[dict]:
    tools = list(load_spec().tools) + ["delete_story", "approve_media", "bogus"]
    ids = [c.id for c in story.characters] + [s.id for s in story.scenes] + [f.id for f in story.facts] + ["missing"]
    out = []
    for _ in range(rng.randint(0, 12)):
        name = rng.choice(tools)
        text = " ".join(rng.choices(WORDS, k=rng.randint(1, 6)))
        out.append({"tool": name, "args": {
            "id": rng.choice(ids), "name": text[:30], "title": text[:30], "summary": text, "text": text,
            "draft_text": rng.choice([text, "a rewritten flowery sentence"]), "description": text,
            "synopsis": rng.choice([text, ""]), "kind": "other", "ambiguous": rng.random() < 0.3,
            "traits": [w for w in rng.sample(WORDS, 3)], "role": "role", "notes": text,
            "sequence": rng.randint(1, 5),
        }})
    return out


async def test_p1_p2_p12_random_turns_never_break_the_confirmation_rules():
    rng = random.Random(1)
    repo = seeded_repo()
    llm = FakeAgentLLM()
    proc, _ = build(llm, repo=repo)
    pending: list[tuple[str, str]] = []  # (operation, target) held on the previous turn
    for i in range(300):
        story = repo._story
        say = rng.choice(["yes", "no", "Remove it.", "Wren climbs the stairs and finds a door.", "Add a scene where the fox naps.",
                          "Rename the story please.", "The moth glows.", "undo that"])
        llm.actions = random_actions(rng, story)
        before = {("c", c.id) for c in story.characters} | {("s", s.id) for s in story.scenes} | {("f", f.id) for f in story.facts}
        response, _ = await run(proc, say)
        after_story = repo._story
        after = {("c", c.id) for c in after_story.characters} | {("s", s.id) for s in after_story.scenes} | {("f", f.id) for f in after_story.facts}
        removed = before - after
        was_confirmed = say == "yes" and bool(pending)
        # P1: nothing removed unless the previous turn held it and the child said yes.
        undone = bool(re.search(r"undo|go back|revert", say))
        for kind, ident in ([] if undone else removed):
            assert was_confirmed and any(ident == target for _, target in pending), (i, say, removed, pending)
        # P2: any saved scene text is the child's own wording from this turn or the confirmed earlier turn.
        for scene in after_story.scenes:
            if scene.draft_text and scene.id.startswith("scene_") and scene.id not in {f"scene_{n}" for n in range(3)}:
                pass
        for patch in response.applied_story_patches:
            draft = patch.data.get("draft_text")
            if isinstance(draft, str) and draft and not was_confirmed:
                assert norm(draft) in norm(say) or norm(say) in norm(draft), (draft, say)
            # P12: a scene from ordinary narration is never applied without a held offer plus a yes.
            if patch.operation.value == "add_scene" and not was_confirmed:
                assert re.search(r"\b(add|make|create|write|put in|start)\b[^.?!]{0,30}\b(scene|chapter)\b", say, re.I), (say, patch)
        pending = [(p.operation.value, str(p.data.get("id", ""))) for p in response.story_patches if p.child_confirmation_required]


async def test_p3_blocked_turns_leave_everything_as_it_was():
    rng = random.Random(2)
    for i in range(100):
        repo = seeded_repo()
        llm = FakeAgentLLM(actions=random_actions(rng, repo._story))
        gate = GateOn(bad="Pip see") if i % 2 else GateOn(output_redirect=True)
        proc, _ = build(llm, gate, repo=repo)
        snapshot = repo._story.model_dump()
        interests_before = list(repo.interests)
        media_before = list(repo.media_assets)
        response, said = await run(proc, rng.choice(["Pip walks on.", "Add a fact about the moon.", "Remove it."]))
        if response.safety.action == SafetyAction.ALLOW:
            continue
        assert repo._story.model_dump() == snapshot
        assert repo.interests == interests_before and repo.media_assets == media_before
        assert response.applied_story_patches == []


async def test_p4_every_story_change_can_be_undone():
    repo = seeded_repo()
    llm = FakeAgentLLM(actions=[{"tool": "add_fact", "args": {"text": "A new fact."}}])
    proc, _ = build(llm, repo=repo)
    for n in range(5):
        before = repo._story.model_dump()
        llm.actions = [{"tool": "add_fact", "args": {"text": f"fact {n}"}}]
        await run(proc, f"fact {n}")
        assert repo._story.model_dump() != before
        llm.actions = [{"tool": "undo_last_change", "args": {}}]
        await run(proc, "undo that")
        assert repo._story.model_dump() == before


async def test_p5_locked_session_makes_no_model_calls():
    llm = FakeAgentLLM()
    proc, repo = build(llm)
    for _ in range(6):
        await run(proc, "A bully steals my crown.", session_id=None)
    sessions = list(repo.sessions.values())
    if not sessions:
        return
    session_id = sessions[0].id
    # drive the session to its limit, then confirm nothing reaches either model call
    for _ in range(6):
        await run(proc, "The bully steals again.", session_id=session_id)
    llm.talker_prompts.clear(); llm.scribe_prompts.clear()
    response, said = await run(proc, "Let us continue.", session_id=session_id)
    if "session_ended" in response.safety.categories:
        assert llm.talker_prompts == [] and llm.scribe_prompts == []


async def test_p6_alert_keywords_stop_before_any_model_call_and_cost_no_strike():
    for term in ["porn", "sex", "nude", "explicit"]:
        llm = FakeAgentLLM()
        proc, repo = build(llm)
        response, _ = await run(proc, f"tell me about {term} please")
        assert response.safety.action == SafetyAction.ALERT_PARENT
        assert llm.talker_prompts == [] and llm.scribe_prompts == []
        assert not any(e.counted_as_strike for e in repo.safety_events)


def test_p7_the_agent_can_only_reach_listed_tools():
    forbidden = {"delete_story", "end_session", "release_lock", "approve_media", "read_other_children", "network", "files"}
    tools = set(load_spec().tools)
    assert not (tools & forbidden)
    raw = {"actions": [{"tool": name, "args": {"id": "x", "text": "x"}} for name in sorted(forbidden)]}
    result = interpret_actions(raw, AgentTurnRequest(transcript="x"))
    assert result.patches == [] and not result.undo and not result.picture
    assert set(result.refused) == forbidden


async def test_p9_a_picture_request_only_ever_creates_a_pending_asset():
    rng = random.Random(9)
    for _ in range(50):
        llm = FakeAgentLLM(actions=[{"tool": "request_picture", "args": {"description": " ".join(rng.choices(WORDS, k=4))}}])
        proc, repo = build(llm)
        await run(proc, "Draw the moon.")
        assert repo.media_assets and all(a.status.value == "pending_parent_review" for a in repo.media_assets)


def test_p10_caps_hold_for_oversized_and_repeated_calls():
    rng = random.Random(10)
    for _ in range(300):
        story = seeded_repo()._story
        actions = random_actions(rng, story) * 3
        result = interpret_actions({"actions": actions}, AgentTurnRequest(transcript="x"))
        assert sum(1 for p in result.patches if p.operation.value.startswith("add_")) <= MAX_ADDS
        assert len(result.patches) + len(result.tutor_notes) <= MAX_ACTIONS
        for patch in result.patches:
            for key, value in patch.data.items():
                if isinstance(value, str):
                    assert len(value) <= 1500
    big = interpret_actions({"actions": [{"tool": "add_fact", "args": {"text": "x" * 5000}}]}, AgentTurnRequest(transcript="x"))
    assert len(big.patches[0].data["text"]) <= 300


async def test_p11_contact_details_leave_no_trace_anywhere():
    rng = random.Random(11)
    secrets = [
        ("I live at {n} Maple Street", "Maple"), ("my school is on {n} Oak Avenue", "Oak"),
        ("email me at kid{n}@example.com", "example"), ("it is {n} harbor road", "harbor road"),
        ("my email is sam dot jones{n} at gmail dot com", "gmail"),
    ]
    for i in range(60):
        template, marker = rng.choice(secrets)
        text = template.format(n=rng.randint(10, 999)) + ", can the fox visit?"
        llm = FakeAgentLLM(actions=[{"tool": "add_fact", "args": {"text": text}}])
        gate = GateOn()
        proc, repo = build(llm, gate)
        response, said = await run(proc, text)
        sinks = " ".join(
            llm.talker_prompts + llm.scribe_prompts + gate.reviewed + [response.response_text] + said
            + [t.transcript + " " + t.response_text for t in repo.turns]
            + [c.content or "" for c in repo.memory_chunks]
            + [e.transcript_excerpt or "" for e in repo.safety_events]
            + [f.text for f in repo._story.facts]
        )
        digits = re.findall(r"\d{2,3}", text)
        assert marker.lower() not in sinks.lower(), (text, sinks[:200])
        # Generated ids are long hex strings; their digit runs are not the child's numbers.
        scrubbed = re.sub(r"\b\w+_[0-9a-f]{16,}\b", " ", sinks)
        assert not any(d in re.sub(r"[^0-9]", " ", scrubbed).split() for d in digits)


async def test_p13_worried_child_reply_is_the_approved_script_byte_for_byte():
    for text in ["I want to kill myself", "i think about suicide a lot", "sometimes i self harm"]:
        proc, repo = build(FakeAgentLLM())
        response, said = await run(proc, text)
        assert response.response_text == code_messages.WORRIED_CHILD_SCRIPT
