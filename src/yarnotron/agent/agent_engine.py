"""The agent-first story engine: a Talker that speaks, a Scribe that proposes, and code that decides.

One turn, in this order:

1. The Talker streams plain spoken text from the constitution. It has no tools and cannot change anything.
2. Each sentence is safety-gated as it is written. The gate for sentence N+1 runs while sentence N is spoken.
   The final sentence is held until the Scribe has answered, because the Scribe's verdict can replace it
   (a confirmation question, a scene offer, a picture or undo note).
3. In parallel from the first token, the Scribe reads the child's words and the story and proposes actions as
   JSON. Code validates every action against `spec/tools.schema.json`, enforces quotes, caps and the confirmation
   classes, and runs the existing editing policy over the resulting patches.
4. Nothing is written until the caller's full output review of the same turn has passed. The caller applies
   `TurnOutcome.apply_patches` only then.

The deterministic guards (keywords, strikes, session lock, parent release, personal-details removal, deferred
review) all live in the caller and run before and after this module. See `agents` notes in `spec/`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from yarnotron.agent import code_messages
from yarnotron.agent.editing_policy import StoryEditingPolicy, is_undo_request
from yarnotron.agent.story_context import compile_story_context
from yarnotron.core.timing import LatencyTimer
from yarnotron.models.agent import AgentTurnRequest, TurnIntent, TutorNote
from yarnotron.models.interests import ChildInterest
from yarnotron.models.memory import MemorySearchResult
from yarnotron.models.safety import SafetyAction, SafetyDecision
from yarnotron.models.story import Story, StoryPatch, StoryPatchOperation
from yarnotron.services.incremental_json import SentenceAccumulator
from yarnotron.services.ollama import OllamaClient, OllamaUnavailableError
from yarnotron.services.pii import redact_contact_details
from yarnotron.services.story_store import StoryRepository
from yarnotron.agent.safety import SafetyReviewer

logger = logging.getLogger(__name__)

SPEC_DIR = Path(__file__).parent / "spec"
SentenceSink = Callable[[str], Awaitable[None]]

_PATCH_TOOLS = {op.value for op in StoryPatchOperation}
_NON_PATCH_TOOLS = {"undo_last_change", "note_interest", "request_picture", "add_tutor_note"}
_PICTURE_WORDS = re.compile(r"\b(picture|image|draw|drawing|paint|painting|illustrat\w*)\b", re.IGNORECASE)
_EXPLICIT_SCENE = re.compile(
    r"\b(add|make|create|write|put in|start)\b[^.?!]{0,30}\b(scene|chapter)\b", re.IGNORECASE
)
# Only the code knows whether something was saved or removed. When the child is asking for a change, a Talker
# sentence that says it already happened is dropped before it is spoken.
_EDIT_WORDS = re.compile(
    r"\b(remove|delete|erase|get rid|take (it |that |him |her |them )?out|cut|undo|change|replace|rename|save|add|"
    r"put (it |that )?(back|in))\b",
    re.IGNORECASE,
)
_CHANGE_CLAIM = re.compile(
    r"(\b(is|are|was|were|has been|have been|got|gets)\s+(now\s+)?"
    r"(gone|removed|deleted|erased|taken out|saved|added|changed|updated|replaced|restored)\b"
    r"|\bi(\s+have|'ve|\u2019ve|'ll|\s+will)?\s+(just\s+|now\s+|already\s+)?"
    r"(removed|deleted|erased|took out|taken out|saved|added|changed|updated|replaced|undid|restored|put back)\b"
    r"|\b(removed|deleted|saved|added) (that|it|this|him|her|them)\b"
    r"|\bno longer in the story\b)",
    re.IGNORECASE,
)
MAX_ACTIONS = 8
MAX_ADDS = 3


# ------------------------------------------------------------------------------------------------ spec


@dataclass(frozen=True)
class AgentSpec:
    talker_rules: str
    scribe_rules: str
    tools: dict[str, dict[str, Any]]


@lru_cache(maxsize=1)
def load_spec() -> AgentSpec:
    text = (SPEC_DIR / "constitution.md").read_text(encoding="utf-8")
    sections: dict[int, str] = {}
    for block in re.split(r"(?m)^## ", text)[1:]:
        match = re.match(r"(\d+)\.\s", block)
        if match:
            sections[int(match.group(1))] = "## " + block.strip()

    def clean(value: str) -> str:
        return re.sub(r"\s*\*\*\[[^\]]*\]\*\*", "", value)

    talker = "\n\n".join(clean(sections[n]) for n in (1, 2, 3, 4, 5, 6, 8, 9) if n in sections)
    scribe = clean(sections.get(7, ""))
    tools_doc = json.loads((SPEC_DIR / "tools.schema.json").read_text(encoding="utf-8"))
    tools = {tool["name"]: tool for tool in tools_doc["tools"]}
    return AgentSpec(talker_rules=talker, scribe_rules=scribe, tools=tools)


def _compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------------------------- prompts


def build_talker_prompt(
    request: AgentTurnRequest,
    story: Story | None,
    memories: list[MemorySearchResult],
    interests: list[ChildInterest],
    spec: AgentSpec | None = None,
) -> str:
    """Constitution first and unchanged, so the model server's prompt cache stays warm."""
    spec = spec or load_spec()
    memory_context = [{"content": m.content, "metadata": m.metadata} for m in memories]
    interest_context = [
        {"name": i.name, "summary": i.summary} if i.summary else {"name": i.name} for i in interests
    ]
    return f"""{spec.talker_rules}

---
Speak the next reply as plain text, ready to be read aloud. No JSON, no lists, no labels.
Current story state (a bounded view; items listed here already exist):
{_compact(compile_story_context(story))}

Relevant prior memory:
{_compact(memory_context)}

Known interests for this child:
{_compact(interest_context)}

Child said: {request.transcript}
"""


def build_scribe_prompt(
    request: AgentTurnRequest,
    story: Story | None,
    interests: list[ChildInterest],
    spec: AgentSpec | None = None,
) -> str:
    spec = spec or load_spec()
    tool_lines = []
    for name, tool in spec.tools.items():
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])
        fields = ", ".join(f"{key}{'*' if key in required else ''}" for key in params)
        tool_lines.append(f"- {name}({fields}) [{tool.get('class', 'auto')}] {tool.get('description', '')}".strip())
    known = [i.name for i in interests]
    return f"""You are the Scribe for Yarnotron. You never speak to the child. Another voice is talking to them
right now. Your only job is to decide which story changes, if any, the child's latest words call for.

Return only valid JSON: {{"actions": [{{"tool": "<name>", "args": {{...}}}}]}}
Most turns need no actions: return {{"actions": []}}. Never invent changes the child did not ask for or add.
Offer a scene (add_scene) only for something big and new, such as a new place or a turning point.
Fields marked * are required. Use ids from the story state when updating or removing.

{spec.scribe_rules}

Tools:
{chr(10).join(tool_lines)}

Current story state (a bounded view):
{_compact(compile_story_context(story))}

Known interests (do not note these again unless the child says something new): {_compact(known)}

Child said: {request.transcript}
"""


# ---------------------------------------------------------------------------------------- scribe output


@dataclass
class ScribeResult:
    patches: list[StoryPatch] = field(default_factory=list)
    undo: bool = False
    picture: bool = False
    interest: tuple[str, bool] | None = None  # (name, ambiguous)
    tutor_notes: list[TutorNote] = field(default_factory=list)
    scene_offer: bool = False
    refused: list[str] = field(default_factory=list)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _clean_args(tool: dict[str, Any], args: object) -> dict[str, Any] | None:
    """Keep only declared fields of the declared type, trim to length. None if a required field is missing."""
    if not isinstance(args, dict):
        args = {}
    schema = tool.get("parameters", {})
    props: dict[str, dict[str, Any]] = schema.get("properties", {})
    out: dict[str, Any] = {}
    for key, spec in props.items():
        if key not in args or args[key] is None:
            continue
        value = args[key]
        kind = spec.get("type")
        if "enum" in spec:
            if value not in spec["enum"]:
                continue
        elif kind == "string":
            if not isinstance(value, str):
                continue
            value = value.strip()[: spec.get("maxLength", 1000)]
        elif kind == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if "minimum" in spec and value < spec["minimum"]:
                continue
        elif kind == "boolean":
            value = bool(value)
        elif kind == "array":
            if not isinstance(value, list):
                continue
            item = spec.get("items", {})
            limit = item.get("maxLength", 200)
            value = [str(v).strip()[:limit] for v in value if str(v).strip()][: spec.get("maxItems", 20)]
        if isinstance(value, str):
            # Defence in depth: the model never sees a removed address, but anything that looks like one is
            # removed again before it can be saved.
            value = redact_contact_details(value).text
        out[key] = value
    for key in schema.get("required", []):
        if key not in out or out[key] in ("", []):
            return None
    return out


def interpret_actions(
    raw: object,
    request: AgentTurnRequest,
    spec: AgentSpec | None = None,
    *,
    allow_scene_offer: bool = True,
) -> ScribeResult:
    """Validate the Scribe's proposed actions. Anything not on the tool list is refused, never executed."""
    spec = spec or load_spec()
    result = ScribeResult()
    actions = raw.get("actions") if isinstance(raw, dict) else None
    if not isinstance(actions, list):
        return result
    transcript_norm = _norm(request.transcript)
    explicit_scene = bool(_EXPLICIT_SCENE.search(request.transcript))
    adds = 0
    for item in actions[:MAX_ACTIONS]:
        name = item.get("tool") if isinstance(item, dict) else None
        tool = spec.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            result.refused.append(str(name))
            continue
        args = _clean_args(tool, item.get("args"))
        if args is None:
            result.refused.append(f"{name}:bad_args")
            continue

        if name == "undo_last_change":
            # Undo is the child's request, never the model's idea.
            if is_undo_request(request.transcript):
                result.undo = True
            else:
                result.refused.append("undo_last_change:not_requested")
        elif name == "request_picture":
            if _PICTURE_WORDS.search(request.transcript):
                result.picture = True
            else:
                result.refused.append("request_picture:not_requested")
        elif name == "note_interest":
            if result.interest is None:
                result.interest = (args["name"], bool(args.get("ambiguous")))
        elif name == "add_tutor_note":
            result.tutor_notes.append(TutorNote(kind=args["kind"], text=args["text"]))
        elif name in _PATCH_TOOLS:
            if name.startswith("add_"):
                adds += 1
                if adds > MAX_ADDS:
                    result.refused.append(f"{name}:cap")
                    continue
            # Story wording must be the child's own: drop a draft that is not a quote of what they said.
            draft = args.get("draft_text")
            if isinstance(draft, str) and _norm(draft) not in transcript_norm:
                args.pop("draft_text")
            flagged = False
            if name == "add_scene" and not explicit_scene:
                if not allow_scene_offer:
                    # A scene was offered or added a moment ago: do not nag.
                    result.refused.append("add_scene:cooldown")
                    continue
                flagged = True
                result.scene_offer = True
            result.patches.append(
                StoryPatch(
                    operation=StoryPatchOperation(name),
                    data=args,
                    child_confirmation_required=flagged,
                    rationale="Scribe proposal"
                    + (" from ordinary narration; the child is asked first." if flagged else "."),
                )
            )
        else:
            result.refused.append(str(name))
    if result.undo:
        # "Undo means undo": no other change rides along.
        result.patches = []
        result.picture = False
        result.interest = None
        result.scene_offer = False
    return result


# ------------------------------------------------------------------------------------------------ outcome


@dataclass
class TurnOutcome:
    reply_text: str
    patches: list[StoryPatch]
    apply_patches: list[StoryPatch]
    undo: bool = False
    picture: bool = False
    interest: tuple[str, bool] | None = None
    tutor_notes: list[TutorNote] = field(default_factory=list)
    intent: TurnIntent = TurnIntent.CONTINUE_STORY
    vetoed: SafetyDecision | None = None
    # What the model wrote, without the code-supplied lead (caution, recap, story intro, not-kept line).
    # The output review sees this, so code's own sentences are never judged as if the model had said them.
    model_text: str = ""
    scribe_failed: bool = False
    refused: list[str] = field(default_factory=list)


def _known_ids(story: Story | None) -> set[str]:
    if story is None:
        return set()
    return {c.id for c in story.characters} | {sc.id for sc in story.scenes} | {f.id for f in story.facts}


def _resolve_unknown_id(patch: StoryPatch, story: Story | None, transcript: str, known: set[str]) -> StoryPatch:
    """A Scribe id that is not in the story is replaced by the one thing the child named, if there is exactly one."""
    current = patch.data.get("id")
    if current is None or current in known or story is None:
        return patch
    operation = patch.operation.value
    if "character" in operation:
        options = [(c.id, c.name) for c in story.characters]
    elif "scene" in operation:
        options = [(sc.id, sc.title) for sc in story.scenes]
    else:
        return patch
    spoken = transcript.casefold()
    hits = [
        item_id for item_id, name in options
        if name and re.search(rf"\b{re.escape(name.casefold())}\b", spoken)
    ]
    if len(hits) != 1:
        return patch
    return patch.model_copy(update={"data": {**patch.data, "id": hits[0]}})


def _drop_trailing_question(text: str) -> str:
    text = text.strip()
    if not text.endswith("?"):
        return text
    head = text[:-1]
    boundary = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
    return head[: boundary + 1].strip() if boundary != -1 else ""


def _interest_question(name: str) -> str:
    return f"Is {name} part of this story, or just something you love?"


class AgentEngine:
    def __init__(
        self,
        llm: OllamaClient,
        safety: SafetyReviewer,
        editing_policy: StoryEditingPolicy | None = None,
        scribe_budget_seconds: float = 6.0,
        spec: AgentSpec | None = None,
    ) -> None:
        self._llm = llm
        self._safety = safety
        self._policy = editing_policy or StoryEditingPolicy()
        self._budget = scribe_budget_seconds
        self._spec = spec or load_spec()

    async def story_turn(
        self,
        request: AgentTurnRequest,
        story: Story | None,
        memories: list[MemorySearchResult],
        interests: list[ChildInterest],
        on_sentence: SentenceSink,
        *,
        prefix: str | None,
        timer: LatencyTimer,
        allow_scene_offer: bool = True,
    ) -> TurnOutcome:
        # The Scribe starts once the first sentence is on its way to the child (or the Talker has finished),
        # not at the first token: both read the same model server, and a prompt being read for the Scribe
        # delays the first word of the Talker.
        holder: dict[str, asyncio.Task[ScribeResult | None]] = {}

        def start_scribe() -> asyncio.Task[ScribeResult | None]:
            if "task" not in holder:
                holder["task"] = asyncio.create_task(
                    self._scribe(request, story, interests, timer, allow_scene_offer)
                )
            return holder["task"]

        spoken: list[str] = []
        try:
            return await self._talk(
                request, story, memories, interests, on_sentence, prefix, timer, start_scribe, spoken
            )
        finally:
            task = holder.get("task")
            if task is not None and not task.done():
                task.cancel()

    # ------------------------------------------------------------------ scribe

    async def _scribe(
        self,
        request: AgentTurnRequest,
        story: Story | None,
        interests: list[ChildInterest],
        timer: LatencyTimer,
        allow_scene_offer: bool = True,
    ) -> ScribeResult | None:
        started = time.perf_counter()
        try:
            prompt = build_scribe_prompt(request, story, interests, self._spec)
            raw = await asyncio.wait_for(
                self._llm.generate_json(prompt, model=request.model, options={"num_predict": 400}, think=False),
                timeout=self._budget,
            )
            return interpret_actions(raw, request, self._spec, allow_scene_offer=allow_scene_offer)
        except (asyncio.TimeoutError, OllamaUnavailableError, ValueError) as exc:
            logger.warning("scribe produced nothing this turn: %s", type(exc).__name__)
            return None
        finally:
            timer.add_metadata(scribe_ms=round((time.perf_counter() - started) * 1000, 1))

    # ------------------------------------------------------------------ talker

    async def _gate(self, request: AgentTurnRequest, sentence: str) -> SafetyDecision:
        return await self._safety.review_assistant_output(
            transcript=request.transcript, response_text=sentence, story_patches=[]
        )

    async def _talk(
        self,
        request: AgentTurnRequest,
        story: Story | None,
        memories: list[MemorySearchResult],
        interests: list[ChildInterest],
        on_sentence: SentenceSink,
        prefix: str | None,
        timer: LatencyTimer,
        start_scribe: Callable[[], asyncio.Task[ScribeResult | None]],
        spoken: list[str],
    ) -> TurnOutcome:
        prompt = build_talker_prompt(request, story, memories, interests, self._spec)
        started = time.perf_counter()
        accumulator = SentenceAccumulator()
        sentences: list[str] = []
        gates: list[asyncio.Task[SafetyDecision]] = []
        state = {"released": 0}  # sentences already spoken; shared with _finish
        veto: SafetyDecision | None = None
        first_token = False

        async def speak_ready(upto: int) -> None:
            """Speak sentences [released, upto) in order once each has passed its gate."""
            nonlocal veto
            while veto is None and state["released"] < upto:
                released = state["released"]
                decision = await gates[released]
                if decision.action != SafetyAction.ALLOW:
                    veto = decision
                    timer.add_metadata(sentence_gate_vetoed=True)
                    return
                if released == 0:
                    timer.add_metadata(
                        model_to_first_audio_ms=round((time.perf_counter() - started) * 1000, 1)
                    )
                await on_sentence(sentences[released])
                spoken.append(sentences[released])
                state["released"] = released + 1
                start_scribe()

        edit_request = bool(_EDIT_WORDS.search(request.transcript))
        claims_dropped = 0

        def add_sentence(text: str) -> None:
            nonlocal claims_dropped
            if not text.strip():
                return
            if edit_request and _CHANGE_CLAIM.search(text):
                claims_dropped += 1
                timer.add_metadata(talker_claims_dropped=claims_dropped)
                return
            sentences.append(text.strip())
            gates.append(asyncio.create_task(self._gate(request, text.strip())))

        if prefix:
            await on_sentence(prefix)
            spoken.append(prefix)

        async for piece in self._llm.stream_text(prompt, model=request.model, options={"num_predict": 160}, think=False):
            if not first_token:
                first_token = True
                timer.add_metadata(model_to_first_token_ms=round((time.perf_counter() - started) * 1000, 1))
            for sentence in accumulator.feed(piece):
                add_sentence(sentence)
                # Speak as soon as a sentence has passed its gate. A sentence ending in a question is probably
                # the last one, which the Scribe's verdict may replace, so it waits for the next sentence or
                # the end of the stream.
                hold = 1 if sentences and sentences[-1].rstrip().endswith("?") else 0
                await speak_ready(len(sentences) - hold)
                if veto is not None:
                    break
            if veto is not None:
                break
        if veto is None:
            add_sentence(accumulator.flush())
            await speak_ready(len(sentences) - 1)

        scribe = await start_scribe()
        timer.add_metadata(talker_sentences=len(sentences), scribe_ok=scribe is not None)

        if veto is not None:
            for task in gates:
                task.cancel()
            message = veto.child_message or "Let's steer that in a safer direction and keep the story adventurous."
            return TurnOutcome(
                reply_text=" ".join(spoken).strip(),
                patches=[],
                apply_patches=[],
                vetoed=veto,
            )

        return await self._finish(request, story, sentences, gates, scribe, on_sentence, prefix, spoken, speak_ready, state, timer)

    async def _finish(
        self,
        request: AgentTurnRequest,
        story: Story | None,
        sentences: list[str],
        gates: list[asyncio.Task[SafetyDecision]],
        scribe: ScribeResult | None,
        on_sentence: SentenceSink,
        prefix: str | None,
        spoken: list[str],
        speak_ready: Callable[[int], Awaitable[None]],
        state: dict[str, int],
        timer: LatencyTimer,
    ) -> TurnOutcome:
        outcome = TurnOutcome(reply_text="", patches=[], apply_patches=[], scribe_failed=scribe is None)
        last = sentences[-1] if sentences else ""
        last_already_spoken = bool(sentences) and state["released"] >= len(sentences)
        replacement: str | None = None
        patches: list[StoryPatch] = []

        if scribe is not None:
            outcome.refused = scribe.refused
            outcome.tutor_notes = list(scribe.tutor_notes)
            if scribe.undo:
                outcome.undo = True
                outcome.intent = TurnIntent.UNDO_LAST_CHANGE
            elif scribe.picture:
                outcome.picture = True
                outcome.intent = TurnIntent.CREATE_IMAGE
                replacement = code_messages.PICTURE_SAVED
            if scribe.patches and not scribe.undo:
                known = _known_ids(story)
                resolved = [_resolve_unknown_id(p, story, request.transcript, known) for p in scribe.patches]
                valid = [p for p in resolved if "id" not in p.data or p.data["id"] in known]
                dropped = [p for p in resolved if p not in valid]
                outcome.refused += [f"{p.operation.value}:unknown_id" for p in dropped]
                if dropped and not valid:
                    # The child asked for a change and code cannot tell which thing they mean: ask, never go quiet.
                    replacement = code_messages.UNCLEAR_TARGET
                reviewed = self._policy.review(valid, story, request.transcript, TurnIntent.CONTINUE_STORY)
                patches = reviewed.patches
                if patches:
                    outcome.intent = TurnIntent.EDIT_STORY
                if reviewed.needs_confirmation or scribe.scene_offer:
                    only_scene_offer = scribe.scene_offer and not reviewed.confirmation_notes
                    replacement = (
                        code_messages.SCENE_OFFER_QUESTION if only_scene_offer else code_messages.CONFIRM_QUESTION
                    )
            if scribe.interest is not None and not scribe.undo:
                name, ambiguous = scribe.interest
                outcome.interest = (name, ambiguous)
                if ambiguous:
                    patches = [p.model_copy(update={"child_confirmation_required": True}) for p in patches]
                    replacement = _interest_question(name)

        if replacement is not None and not outcome.undo:
            # The last sentence normally has not been spoken yet and gives way to the code's line. If it was
            # already spoken (no closing question), the line follows it.
            final = replacement if last_already_spoken else f"{_drop_trailing_question(last)} {replacement}".strip()
            if gates and not last_already_spoken:
                gates[-1].cancel()
            decision = await self._gate(request, final)
            if decision.action != SafetyAction.ALLOW:
                outcome.vetoed = decision
                outcome.reply_text = " ".join(spoken).strip()
                outcome.model_text = " ".join(spoken[1:] if prefix else spoken).strip()
                return outcome
            await on_sentence(final)
            spoken.append(final)
        elif outcome.undo:
            # The caller says the result line once it knows whether there was anything to undo.
            pass
        else:
            await speak_ready(len(sentences))
            if self._vetoed_in(gates):
                outcome.vetoed = self._vetoed_in(gates)
                outcome.reply_text = " ".join(spoken).strip()
                outcome.model_text = " ".join(spoken[1:] if prefix else spoken).strip()
                return outcome

        outcome.patches = patches
        outcome.apply_patches = [p for p in patches if not p.child_confirmation_required]
        outcome.reply_text = " ".join(spoken).strip()
        outcome.model_text = " ".join(spoken[1:] if prefix else spoken).strip()
        return outcome

    @staticmethod
    def _vetoed_in(gates: list[asyncio.Task[SafetyDecision]]) -> SafetyDecision | None:
        for task in gates:
            if task.done() and not task.cancelled() and task.result().action != SafetyAction.ALLOW:
                return task.result()
        return None
