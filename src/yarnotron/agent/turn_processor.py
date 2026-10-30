import asyncio
import contextvars
from datetime import UTC, datetime
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeAlias

from fastapi import Depends
from pydantic import BaseModel, ValidationError

from yarnotron.agent import code_messages
from yarnotron.agent import turn_acts as acts
from yarnotron.agent.agent_engine import AgentEngine
from yarnotron.agent.editing_policy import StoryEditingPolicy, StoryEditingPolicyResult
from yarnotron.agent.prompts import build_story_turn_prompt
from yarnotron.agent.router import IntentRouter
from yarnotron.agent.safety import SafetyReviewer, combine_safety_decisions, get_safety_reviewer
from yarnotron.core.config import Settings, get_settings
from yarnotron.core.timing import LatencyTimer
from yarnotron.models.agent import AgentDraft, AgentTurnRequest, AgentTurnResponse, TurnIntent, TutorNote
from yarnotron.models.interests import ChildInterest, InterestCandidate
from yarnotron.models.media import CreateMediaAssetRequest
from yarnotron.models.memory import ConversationTurn
from yarnotron.models.operations import (
    STRIKE_KEY,
    CreateSessionRequest,
    SafetyEvent,
    StorySession,
)
from yarnotron.services.sentences import drop_trailing_question, split_sentences
from yarnotron.services.transcript_clean import strip_sound_tags
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import CreateStoryRequest, Story, StoryPatch
from yarnotron.services.incremental_json import SentenceAccumulator
from yarnotron.services.interests import InterestEnricher, schedule_enrichment
from yarnotron.services.pii import PII_NOT_KEPT_LINE, redact_contact_details
from yarnotron.services.memory import (
    Embedder,
    StoryMemoryBuilder,
    embed_document,
    embed_query,
    get_embedder,
    get_memory_builder,
)
from yarnotron.services.shutdown import get_shutdown
from yarnotron.services.ollama import OllamaClient, OllamaUnavailableError, get_ollama_client
from yarnotron.services.story_store import StoryRepository, get_story_repository

logger = logging.getLogger(__name__)

# Called with each safety-gated sentence as soon as it is ready to speak.
SentenceSink: TypeAlias = Callable[[str], Awaitable[None]]

# Turns of a story during which a scene is not offered again after one was offered or added.
_SCENE_OFFER_COOLDOWN_TURNS = 4

# The line the child hears when contact details were removed from what they said. Set per turn.
_PII_LEAD: contextvars.ContextVar[str | None] = contextvars.ContextVar("pii_lead", default=None)


class TurnProcessor:
    def __init__(
        self,
        ollama: OllamaClient,
        repository: StoryRepository,
        embedder: Embedder,
        memory_builder: StoryMemoryBuilder,
        safety: SafetyReviewer,
        router: IntentRouter | None = None,
        editing_policy: StoryEditingPolicy | None = None,
        interest_enricher: InterestEnricher | None = None,
        strike_limit: int = 5,
        engine: AgentEngine | None = None,
    ) -> None:
        self._ollama = ollama
        self._repository = repository
        self._embedder = embedder
        self._memory_builder = memory_builder
        self._safety = safety
        self._router = router or IntentRouter()
        self._editing_policy = editing_policy or StoryEditingPolicy()
        self._interest_enricher = interest_enricher
        self._strike_limit = strike_limit
        # None keeps the classic pipeline. See agent/agent_engine.py.
        self._engine = engine
        # Holds the background interest lookups. See services/interests.
        self._interest_tasks: set[asyncio.Task] = set()
        # Deferred input reviews. See _schedule_input_review.
        self._input_review_tasks: set[asyncio.Task] = set()

    async def process(
        self,
        request: AgentTurnRequest,
        *,
        on_sentence: SentenceSink | None = None,
        parent_released: bool = False,
    ) -> AgentTurnResponse:
        """Run one turn.

        Passing `on_sentence` streams the spoken reply: each sentence is safety
        gated and handed over as soon as the model has written it, rather than
        after the whole draft, the editing policy and the full output review.
        The returned response is identical either way -- persistence and patch
        application still happen at the end, which now lands after first audio
        instead of before it.
        """
        timer = LatencyTimer(
            "agent_turn",
            child_id=request.child_id,
            story_id=request.story_id,
            session_id=request.session_id,
            device_id=request.device_id,
        )
        # Contact details are removed before anything else sees the words: no model call, no storage, no parent
        # note ever holds them. The child is told once, and the story carries on.
        pii = redact_contact_details(request.transcript)
        pii_line: str | None = None
        if pii.found:
            request = request.model_copy(update={"transcript": pii.text})
            pii_line = PII_NOT_KEPT_LINE
            timer.add_metadata(pii_removed=",".join(pii.kinds))
        # Background sound that speech recognition wrote down (*thud*, [music]) is not the child talking.
        heard = strip_sound_tags(request.transcript)
        sound_only = not heard
        if heard != request.transcript.strip() and heard:
            request = request.model_copy(update={"transcript": heard})
            timer.add_metadata(sound_tags_removed=True)
        lead_token = _PII_LEAD.set(pii_line)
        try:
            if sound_only:
                timer.add_metadata(sound_only=True)
                return await self._persist_turn(
                    request,
                    AgentTurnResponse(
                        intent=TurnIntent.UNKNOWN,
                        response_text=code_messages.NOT_HEARD,
                        safety=SafetyDecision(action=SafetyAction.ALLOW, severity=SafetySeverity.NONE),
                    ),
                    timer=timer,
                )
            # Before the safety model call, which is ~900 ms: a session that has
            # already run out of strikes should cost nothing to turn away.
            with timer.measure("session_lookup"):
                session = await self._resolve_session(request)
            if session is not None:
                # A caller that sent no session id still gets one, so strikes
                # have somewhere to live and the reply can hand it back.
                request = request.model_copy(update={"session_id": session.id})
                # A parent releasing a block is re-running a turn on purpose,
                # and may well be doing it after the session ended.
                if session.ended_at is not None and not parent_released:
                    return await self._session_already_ended(request, timer=timer)

            if session is not None and _is_filler(request.transcript):
                recent = await self._repository.list_conversation_turns(limit=IDLE_FILLER_LIMIT * 2)
                if 1 + _filler_run(recent, session.id) >= IDLE_FILLER_LIMIT:
                    return await self._end_session_for_idle(request, timer=timer)

            history = await self._session_history(session)
            held = _held_fragment(history)
            if held:
                # The last turn stopped mid-thought and was answered with a nod; this one finishes it.
                request = request.model_copy(update={"transcript": f"{held} {request.transcript}".strip()[:1500]})
                timer.add_metadata(fragment_joined=True)

            # The story is loaded before the safety review, not after, so the
            # reviewer can tell who is in it. A bare sentence gets misread: see
            # build_story_safety_context. The lookup is ~3 ms against a ~900 ms
            # review, and a missing story now fails before the model call.
            story = None
            story_selection = _StorySelection()
            if request.story_id is not None:
                with timer.measure("story_lookup"):
                    story = await self._repository.get(request.story_id)
                if story is None:
                    raise KeyError(request.story_id)

            # One read, two questions: has a parent already let this exact line
            # through in this session, and is there a released turn the child
            # was not around to hear about?
            with timer.measure("release_lookup"):
                events = await self._repository.list_safety_events(limit=50)
            released = parent_released or _transcript_was_released(events, request)
            pending_recap = _pending_recap(events, request.story_id)

            # Routed before the review, not after: it costs nothing and it tells
            # the reviewer whether the child is editing their draft. "Get rid of
            # the speaker, never mention it again" is an edit, and was read as
            # harassment without it.
            with timer.measure("intent_route"):
                intent = self._router.route(request.transcript)
            timer.add_metadata(intent=intent.value)

            # Resolved here rather than at the branch below because whether this
            # names one of the child's stories is the difference between a turn
            # that answers a picker and one that writes. It is an indexed lookup
            # on a handful of rows, and the answer is reused there.
            picked_story = None
            if request.story_id is not None and _looks_like_story_picker_answer(request.transcript):
                with timer.measure("story_switch_lookup"):
                    picked_story = await _find_referenced_story(
                        self._repository, request.transcript
                    )

            # The keyword guard is instant and stops the turn outright on the
            # things that must never wait: self-harm, sexual content. The model
            # reviewer is ~370 ms and used to sit here in front of the child; it
            # now runs after the turn, while Yarnotron is speaking and the GPU is
            # otherwise idle. It can no longer stop the turn -- what it still
            # does is raise the parent alert and spend the strike. See
            # _schedule_input_review.
            if released:
                timer.add_metadata(parent_released=True)
                safety = SafetyDecision(categories=["parent_released"])
            else:
                with timer.measure("input_keywords"):
                    safety = self._safety.classify_transcript(request.transcript)

            stopped, caution = await self._input_safety_outcome(
                request, safety, intent, session, timer=timer
            )
            if stopped is not None:
                return stopped
            # What is spoken and saved in front of the reply. A strike is still spent on `caution` alone.
            spoken_lead = " ".join(part for part in (pii_line, caution) if part) or None

            # A destructive edit asked for confirmation on the previous turn and
            # was deliberately not applied. Resolve it before anything else --
            # "yes" and "no" route to UNKNOWN otherwise, so the child could never
            # actually confirm and the change was silently dropped forever.
            answer = _confirmation_answer(request.transcript)
            if answer is not None and request.story_id is not None:
                with timer.measure("pending_lookup"):
                    previous = await self._previous_turn(request)
                awaiting = _patches_awaiting_confirmation(previous)
                pending_interest = _interest_awaiting_clarification(previous)
                if awaiting or pending_interest is not None:
                    return await self._resolve_confirmation(
                        request,
                        safety,
                        story,
                        awaiting,
                        accepted=answer,
                        timer=timer,
                        interest_name=pending_interest,
                    )

            # "Yes, and he runs to first base" answers a scene offer and carries on. Save the scene, then
            # treat the whole line as the next turn so the rest of it is not lost.
            carried: list[StoryPatch] = []
            if (
                answer is None
                and self._engine is not None
                and request.story_id is not None
                and _leads_with_yes(request.transcript)
            ):
                offered = _patches_awaiting_confirmation(await self._previous_turn(request))
                if offered and all(patch.operation.value == "add_scene" for patch in offered):
                    for patch in offered:
                        confirmed = patch.model_copy(update={"child_confirmation_required": False})
                        try:
                            story = await self._repository.apply_patch(request.story_id, confirmed)
                        except (ValueError, KeyError):
                            logger.warning("dropped a confirmed scene offer that could not be applied")
                            continue
                        carried.append(confirmed)

            act = await self._conversation_act(request, session, story, safety, history, timer)
            if act is not None:
                return act

            if _asks_for_story_list(request.transcript):
                return await self._list_available_stories(request, safety, intent, timer=timer)

            if picked_story is not None:
                if picked_story.id != request.story_id:
                    request = request.model_copy(update={"story_id": picked_story.id})
                    if request.session_id is not None:
                        with timer.measure("session_story_attach"):
                            await self._repository.attach_session_story(
                                request.session_id,
                                picked_story.id,
                            )
                response = _picked_story_response(intent, safety, picked_story)
                return await self._persist_turn(request, response, timer=timer)

            if intent == TurnIntent.UNKNOWN and self._engine is None:
                response = AgentTurnResponse(
                    intent=intent,
                    response_text=(
                        "I did not quite catch how that changes the story. "
                        "Tell me the part you want to add, change, or explore."
                    ),
                    safety=safety,
                    tutor_notes=[
                        TutorNote(
                            kind="clarification",
                            text="The child gave an unclear or non-story request.",
                        )
                    ],
                )
                return await self._persist_turn(request, response, timer=timer)

            if intent == TurnIntent.UNDO_LAST_CHANGE and self._engine is None:
                return await self._undo_last_change(request, safety, story, timer=timer)

            if intent == TurnIntent.CREATE_IMAGE and self._engine is None:
                return await self._request_media_asset(request, safety, timer=timer)

            if request.story_id is None:
                with timer.measure("story_select"):
                    story_selection = await self._select_story_for_first_turn(request, intent)
                if story_selection.needs_clarification:
                    response = AgentTurnResponse(
                        intent=intent,
                        response_text=(
                            "Is this a new story, or are we continuing one we already started? "
                            "If we are continuing, tell me the title."
                        ),
                        safety=safety,
                        tutor_notes=[
                            TutorNote(
                                kind="story_selection",
                                text=(
                                    "The child spoke without an active story and the turn sounded "
                                    "like it might refer to a previous story."
                                ),
                            )
                        ],
                    )
                    return await self._persist_turn(request, response, timer=timer)
                if story_selection.story is not None:
                    story = story_selection.story
                    request = request.model_copy(update={"story_id": story.id})
                    timer.add_metadata(
                        story_id=story.id,
                        story_selection=story_selection.selection,
                    )
                    if request.session_id is not None:
                        with timer.measure("session_story_attach"):
                            await self._repository.attach_session_story(
                                request.session_id,
                                story.id,
                            )
                    if story_selection.picked:
                        response = _picked_story_response(intent, safety, story)
                        return await self._persist_turn(request, response, timer=timer)

            relevant_memories = []
            if request.story_id is not None:
                with timer.measure("memory_query_embedding"):
                    query_embedding = await embed_query(self._embedder, request.transcript)
                with timer.measure("memory_search"):
                    relevant_memories = await self._repository.search_memory(
                        query_embedding=query_embedding,
                        story_id=request.story_id,
                        limit=5,
                    )
                timer.add_metadata(retrieved_memory_count=len(relevant_memories))

            with timer.measure("interest_lookup"):
                interests = await self._repository.list_child_interests(request.child_id)
            timer.add_metadata(interest_count=len(interests))

            # Not on the release re-run itself -- nobody is listening to that
            # one. The recap is for whenever the child next opens the story.
            recap = (
                _recap_message(pending_recap.transcript_excerpt)
                if pending_recap is not None and not parent_released
                else None
            )

            if self._engine is not None:
                return await self._agent_story_turn(
                    request,
                    safety,
                    story,
                    spoken_lead=spoken_lead,
                    caution=caution,
                    recap=recap,
                    pending_recap=pending_recap,
                    carried=carried,
                    story_selection=story_selection,
                    memories=relevant_memories,
                    interests=interests,
                    on_sentence=on_sentence,
                    parent_released=parent_released,
                    timer=timer,
                )

            with timer.measure("prompt_build"):
                prompt = build_story_turn_prompt(
                    request,
                    intent,
                    story=story,
                    relevant_memories=relevant_memories,
                    interests=interests,
                )
            with timer.measure("model_generate"):
                if on_sentence is None:
                    draft_payload = await self._ollama.generate_json(prompt, model=request.model)
                else:
                    draft_payload = await self._stream_draft(
                        prompt,
                        request,
                        on_sentence,
                        timer=timer,
                        prefix=_spoken_prefix(spoken_lead, recap, story_selection),
                    )
            if recap is not None:
                # Said by now on the streaming path, and in the saved text on
                # the buffered one. Either way the child has been told once.
                with timer.measure("recap_mark"):
                    await self._repository.mark_safety_event_recapped(pending_recap.id)
                timer.add_metadata(released_recap_delivered=True)
            with timer.measure("draft_validate"):
                draft = _validate_draft(draft_payload, timer=timer)
            with timer.measure("editing_policy"):
                editing_result = self._editing_policy.review(
                    draft.story_patches,
                    story,
                    request.transcript,
                    intent,
                )
                response_text = _with_confirmation_prompt(draft.response_text, editing_result)
                if request.story_id is not None and story_selection.created:
                    response_text = _with_new_story_intro(response_text, story_selection.story)
                elif request.story_id is not None and story_selection.resumed:
                    response_text = _with_resumed_story_intro(response_text, story_selection.story)
                # Same order the child heard it in: caution, story intro, recap,
                # then the reply. Applied outermost-last so caution leads.
                response_text = _with_recap(response_text, recap)
                response_text = _with_caution(response_text, spoken_lead)
            timer.add_metadata(proposed_patch_count=len(editing_result.patches))

            # An outside interest the child mentioned. When the model could not
            # tell whether it was story content, hold the patches back and ask
            # rather than guessing -- the same deferral the editing policy uses
            # for destructive edits, so one "yes" or "no" next turn settles it.
            patches = editing_result.patches
            interest_notes: list[TutorNote] = []
            candidate = draft.interest_candidate
            if candidate is not None and candidate.ambiguous:
                patches = [
                    patch.model_copy(update={"child_confirmation_required": True})
                    for patch in patches
                ]
                response_text = await self._ask_where_interest_belongs(
                    response_text,
                    candidate.name,
                    request=request,
                    on_sentence=on_sentence,
                )
                interest_notes = [
                    TutorNote(kind=_INTEREST_CLARIFICATION, text=candidate.name)
                ]
                timer.add_metadata(interest_clarification=candidate.name)
            elif candidate is not None:
                with timer.measure("interest_record"):
                    await self._record_interest(request.child_id, candidate.name)
                timer.add_metadata(interest_recorded=candidate.name)

            with timer.measure("output_safety"):
                output_safety = await self._safety.review_assistant_output(
                    transcript=request.transcript,
                    response_text=response_text,
                    story_patches=patches,
                )
            safety = combine_safety_decisions(safety, output_safety)

            # These gate on the *output* review alone, not the combined decision.
            # A cautioned input is carried in `safety` so the parent sees it, and
            # testing the combination here would throw the turn away again --
            # exactly the cutting-off the caution exists to avoid.
            if output_safety.action in _STOP_ACTIONS:
                response = AgentTurnResponse(
                    intent=intent,
                    response_text=self._stop_message(output_safety),
                    safety=safety,
                    retrieved_memory_ids=[memory.id for memory in relevant_memories],
                )
                return await self._persist_turn(request, response, timer=timer)

            if output_safety.action == SafetyAction.REDIRECT:
                response = AgentTurnResponse(
                    intent=intent,
                    response_text=output_safety.child_message
                    or "Let's steer that in a safer direction and keep the story adventurous.",
                    safety=safety,
                    retrieved_memory_ids=[memory.id for memory in relevant_memories],
                )
                return await self._persist_turn(request, response, timer=timer)

            applied_patches = []
            updated_story = story
            if request.story_id is not None:
                with timer.measure("patch_apply"):
                    for patch in patches:
                        if patch.child_confirmation_required:
                            continue
                        updated_story = await self._repository.apply_patch(request.story_id, patch)
                        applied_patches.append(patch)
            timer.add_metadata(applied_patch_count=len(applied_patches))

            response = AgentTurnResponse(
                intent=intent,
                response_text=response_text or _fallback_response(intent),
                safety=safety,
                story_patches=patches,
                applied_story_patches=applied_patches,
                updated_story=updated_story,
                tutor_notes=(
                    draft.tutor_notes or _build_tutor_notes(request.transcript, intent)
                )
                + interest_notes,
                retrieved_memory_ids=[memory.id for memory in relevant_memories],
            )
            # A cautioned turn spends a strike even though the story went on.
            return await self._persist_turn(
                request, response, timer=timer, strike=caution is not None
            )
        finally:
            _PII_LEAD.reset(lead_token)
            timer.log()

    def _stop_message(self, decision: SafetyDecision) -> str:
        """The stop line. With the agent engine a worried child hears the fixed, approved script."""
        if self._engine is not None and _is_worried_child(decision):
            return code_messages.WORRIED_CHILD_SCRIPT
        return _safety_stop_message(decision)

    def _gate_line(self, decision: SafetyDecision, caution: str | None) -> str:
        """What the child hears when the gate stops the reply: fixed code wording, never the model's own.

        A redirect that follows a caution already spoken says nothing more, so the line is not said twice.
        """
        if decision.action in _STOP_ACTIONS:
            return self._stop_message(decision)
        return "" if caution else _DEFAULT_CAUTION

    async def _agent_story_turn(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        story: Story | None,
        *,
        spoken_lead: str | None,
        caution: str | None,
        recap: str | None,
        pending_recap: SafetyEvent | None,
        carried: list[StoryPatch],
        story_selection: "_StorySelection",
        memories: list,
        interests: list[ChildInterest],
        on_sentence: SentenceSink | None,
        parent_released: bool,
        timer: LatencyTimer,
    ) -> AgentTurnResponse:
        """One story turn through the agent engine. Writes happen only after the output review passes."""
        assert self._engine is not None
        streamed = 0

        async def sink(sentence: str) -> None:
            nonlocal streamed
            if on_sentence is not None:
                await on_sentence(sentence)
                streamed += 1

        async def say(line: str) -> None:
            # A line the child has not heard yet. If nothing streamed, the final text is spoken instead.
            if streamed:
                await sink(line)

        memory_ids = [memory.id for memory in memories]
        prefix = _spoken_prefix(spoken_lead, recap, story_selection)
        # No scene offer if a scene was offered or added in the last few turns of this story.
        same_story = [
            turn
            for turn in await self._repository.list_conversation_turns(limit=10)
            if turn.story_id == request.story_id
            and (request.session_id is None or turn.session_id == request.session_id)
        ]
        recent = same_story[:_SCENE_OFFER_COOLDOWN_TURNS]
        quiet = _in_dictation(same_story, request.transcript)
        allow_scene_offer = not any(
            patch.operation.value == "add_scene" for turn in recent for patch in turn.story_patches
        )
        with timer.measure("agent_engine"):
            outcome = await self._engine.story_turn(
                request,
                story,
                memories,
                interests,
                sink,
                prefix=prefix,
                timer=timer,
                allow_scene_offer=allow_scene_offer,
                avoid_questions=_recent_questions(recent),
                quiet=quiet,
            )
        if recap is not None:
            with timer.measure("recap_mark"):
                await self._repository.mark_safety_event_recapped(pending_recap.id)
            timer.add_metadata(released_recap_delivered=True)
        timer.add_metadata(
            proposed_patch_count=len(outcome.patches),
            scribe_refused=len(outcome.refused),
            scribe_failed=outcome.scribe_failed,
        )

        # The gate vetoed a sentence: what was already said stands, the rest is dropped, nothing is saved.
        if outcome.vetoed is not None:
            safety = combine_safety_decisions(safety, outcome.vetoed)
            message = self._gate_line(outcome.vetoed, caution)
            if message:
                await say(message)
            response = AgentTurnResponse(
                intent=outcome.intent,
                response_text=message or outcome.reply_text,
                safety=safety,
                retrieved_memory_ids=memory_ids,
            )
            return await self._persist_turn(request, response, timer=timer, strike=caution is not None)

        with timer.measure("output_safety"):
            output_safety = await self._safety.review_assistant_output(
                transcript=request.transcript,
                response_text=outcome.model_text,
                story_patches=outcome.patches,
            )
        safety = combine_safety_decisions(safety, output_safety)
        if output_safety.action in _STOP_ACTIONS or output_safety.action == SafetyAction.REDIRECT:
            message = self._gate_line(output_safety, caution)
            if message:
                await say(message)
            response = AgentTurnResponse(
                intent=outcome.intent,
                response_text=message or outcome.reply_text,
                safety=safety,
                retrieved_memory_ids=memory_ids,
            )
            return await self._persist_turn(request, response, timer=timer, strike=caution is not None)

        # Passed. Only now does anything change.
        reply = outcome.reply_text
        updated_story = story
        applied: list[StoryPatch] = list(carried)
        media_ids: list[str] = []
        notes = list(outcome.tutor_notes)
        if quiet:
            notes.append(TutorNote(kind=acts.DICTATION_NOTE, text="The child is telling the story; replies stay short."))
            timer.add_metadata(dictation=True)

        if outcome.undo and request.story_id is not None:
            with timer.measure("version_lookup"):
                versions = await self._repository.list_versions(request.story_id)
            if versions:
                with timer.measure("version_restore"):
                    restored = await self._repository.restore_version(request.story_id, versions[0].id)
                updated_story = restored.restored_story
                line = code_messages.UNDO_DONE
                notes.append(TutorNote(kind="editing", text="Explicit undo restored the most recent saved story version."))
            else:
                line = code_messages.UNDO_NOTHING
            await sink(line)
            reply = f"{reply} {line}".strip()
        elif outcome.undo:
            line = "I can help with that once we have a story open. Tell me the story title you want to open."
            await sink(line)
            reply = f"{reply} {line}".strip()
        elif request.story_id is not None:
            with timer.measure("patch_apply"):
                for patch in outcome.apply_patches:
                    try:
                        updated_story = await self._repository.apply_patch(request.story_id, patch)
                    except (ValueError, KeyError):
                        # A proposal that no longer fits the story is dropped; the turn still completes.
                        logger.warning("dropped a scribe patch that could not be applied: %s", patch.operation.value)
                        continue
                    applied.append(patch)
        timer.add_metadata(applied_patch_count=len(applied))

        if outcome.interest is not None and not outcome.undo:
            name, ambiguous = outcome.interest
            if ambiguous:
                notes.append(TutorNote(kind=_INTEREST_CLARIFICATION, text=name))
                timer.add_metadata(interest_clarification=name)
            else:
                with timer.measure("interest_record"):
                    await self._record_interest(request.child_id, name)
                timer.add_metadata(interest_recorded=name)

        if outcome.picture:
            notes.append(
                TutorNote(kind="media", text="Image generation request was saved for parent approval before generation.")
            )

        response = AgentTurnResponse(
            intent=outcome.intent,
            response_text=reply or _fallback_response(outcome.intent),
            safety=safety,
            story_patches=[*carried, *outcome.patches] if not outcome.undo else carried,
            applied_story_patches=applied,
            updated_story=updated_story,
            tutor_notes=notes or _build_tutor_notes(request.transcript, outcome.intent),
            retrieved_memory_ids=memory_ids,
        )
        response = await self._persist_turn(request, response, timer=timer, strike=caution is not None)
        if outcome.picture:
            with timer.measure("media_asset_create"):
                asset = await self._repository.create_media_asset(
                    CreateMediaAssetRequest(
                        prompt=request.transcript,
                        child_id=request.child_id,
                        story_id=request.story_id,
                        session_id=request.session_id,
                        conversation_turn_id=response.turn_id,
                    ),
                    safety=safety,
                )
            media_ids = [asset.id]
            response = response.model_copy(update={"media_asset_ids": media_ids})
        return response

    async def _list_available_stories(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        intent: TurnIntent,
        timer: LatencyTimer | None = None,
    ) -> AgentTurnResponse:
        if timer is None:
            stories = await self._repository.list()
        else:
            with timer.measure("story_list"):
                stories = await self._repository.list()
            timer.add_metadata(available_story_count=len(stories))

        if stories:
            titles = "; ".join(story.title for story in stories)
            response_text = f"Available stories: {titles}. Which one would you like to pick?"
        else:
            response_text = (
                "I do not see any saved stories yet. "
                "Tell me the title for a new story."
            )
        response = AgentTurnResponse(
            intent=intent,
            response_text=response_text,
            safety=safety,
            tutor_notes=[
                TutorNote(
                    kind="story_selection",
                    text="Listed saved story titles for the child to choose from.",
                )
            ],
        )
        return await self._persist_turn(request, response, timer=timer)

    async def _select_story_for_first_turn(
        self,
        request: AgentTurnRequest,
        intent: TurnIntent,
    ) -> "_StorySelection":
        if intent not in {TurnIntent.CONTINUE_STORY, TurnIntent.EDIT_STORY}:
            return _StorySelection()

        existing_story = await _find_referenced_story(self._repository, request.transcript)
        if existing_story is not None:
            picked = _story_matches_picker_response(existing_story, request.transcript)
            return _StorySelection(
                story=existing_story,
                resumed=not picked,
                picked=picked,
                selection="picked" if picked else "resumed",
            )

        if _sounds_like_ambiguous_continuation(request.transcript):
            return _StorySelection(needs_clarification=True, selection="clarify")

        title = _first_turn_story_title(request.transcript)
        story = await self._repository.create(
            CreateStoryRequest(
                title=title,
                synopsis=f"Started from the first spoken turn: {request.transcript.strip()}",
            )
        )
        return _StorySelection(story=story, created=True, selection="created")

    async def _request_media_asset(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        timer: LatencyTimer | None = None,
    ) -> AgentTurnResponse:
        response = AgentTurnResponse(
            intent=TurnIntent.CREATE_IMAGE,
            response_text=(
                f"Oh, that would make a cool picture: {_short_acknowledgement(request.transcript)}. "
                "I saved it for parent review, and we can keep writing while it waits."
            ),
            safety=safety,
            tutor_notes=[
                TutorNote(
                    kind="media",
                    text="Image generation request was saved for parent approval before generation.",
                )
            ],
        )
        persisted = await self._persist_turn(request, response, timer=timer)
        if timer is None:
            media_asset = await self._repository.create_media_asset(
                CreateMediaAssetRequest(
                    prompt=request.transcript,
                    child_id=request.child_id,
                    story_id=request.story_id,
                    session_id=request.session_id,
                    conversation_turn_id=persisted.turn_id,
                ),
                safety=safety,
            )
        else:
            with timer.measure("media_asset_create"):
                media_asset = await self._repository.create_media_asset(
                    CreateMediaAssetRequest(
                        prompt=request.transcript,
                        child_id=request.child_id,
                        story_id=request.story_id,
                        session_id=request.session_id,
                        conversation_turn_id=persisted.turn_id,
                    ),
                    safety=safety,
                )
        return persisted.model_copy(update={"media_asset_ids": [media_asset.id]})

    async def _undo_last_change(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        story: Story | None,
        timer: LatencyTimer | None = None,
    ) -> AgentTurnResponse:
        if request.story_id is None:
            response = AgentTurnResponse(
                intent=TurnIntent.UNDO_LAST_CHANGE,
                response_text=(
                    "I can help with that once we have a story open. "
                    "Tell me the story title you want to open."
                ),
                safety=safety,
            )
            return await self._persist_turn(request, response, timer=timer)

        if timer is None:
            versions = await self._repository.list_versions(request.story_id)
        else:
            with timer.measure("version_lookup"):
                versions = await self._repository.list_versions(request.story_id)
        if not versions:
            response = AgentTurnResponse(
                intent=TurnIntent.UNDO_LAST_CHANGE,
                response_text=(
                    "I do not see a recent story change to undo yet. "
                    "What tiny change should we try next?"
                ),
                safety=safety,
                updated_story=story,
            )
            return await self._persist_turn(request, response, timer=timer)

        if timer is None:
            restored = await self._repository.restore_version(request.story_id, versions[0].id)
        else:
            with timer.measure("version_restore"):
                restored = await self._repository.restore_version(request.story_id, versions[0].id)
        response = AgentTurnResponse(
            intent=TurnIntent.UNDO_LAST_CHANGE,
            response_text=(
                "No problem. I put the story back the way it was before the last change. "
                "What would you like to try instead?"
            ),
            safety=safety,
            updated_story=restored.restored_story,
            tutor_notes=[
                TutorNote(
                    kind="editing",
                    text="Explicit undo restored the most recent saved story version.",
                )
            ],
        )
        return await self._persist_turn(request, response, timer=timer)

    def _schedule_input_review(
        self,
        request: AgentTurnRequest,
        turn_id: str,
        intent: TurnIntent,
    ) -> None:
        """Run the model's view of what the child said, after the turn.

        The child is listening to their reply by now, so the GPU is free and this
        costs them nothing. It cannot stop the turn any more -- the keyword guard
        is what does that, inline -- but the parent alert and the strike still
        land, a few hundred milliseconds later than they used to.
        """

        async def run() -> None:
            try:
                story = (
                    await self._repository.get(request.story_id)
                    if request.story_id is not None
                    else None
                )
                decision = await self._safety.review_transcript(
                    request.transcript, story, intent.value
                )
                if (
                    decision.action == SafetyAction.ALLOW
                    and decision.severity == SafetySeverity.NONE
                ):
                    return
                await self._repository.save_safety_event(
                    SafetyEvent(
                        child_id=request.child_id,
                        story_id=request.story_id,
                        session_id=request.session_id,
                        conversation_turn_id=turn_id,
                        safety=decision,
                        transcript_excerpt=request.transcript[:500],
                        parent_alert_sent=decision.action in _STOP_ACTIONS,
                        counted_as_strike=decision.action == SafetyAction.REDIRECT,
                    )
                )
            except Exception:
                # A background review must never surface as a failed story turn.
                logger.exception("deferred input review failed for turn %s", turn_id)

        task = asyncio.create_task(run())
        get_shutdown().track_task(task)
        self._input_review_tasks.add(task)
        task.add_done_callback(self._input_review_tasks.discard)

    async def _input_safety_outcome(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        intent: TurnIntent,
        session: StorySession | None,
        *,
        timer: LatencyTimer,
    ) -> tuple[AgentTurnResponse | None, str | None]:
        """What the input review means for this turn: stop it, or caution it.

        One place, because the story path now reaches it later than the others
        and the two must not drift apart. Returns the response that ends the
        turn, or the caution to put in front of the reply.
        """
        if safety.action in _STOP_ACTIONS:
            response = AgentTurnResponse(
                intent=intent,
                response_text=self._stop_message(safety),
                safety=safety,
            )
            return await self._persist_turn(request, response, timer=timer), None

        # A milder flag no longer ends the turn. The child hears a caution, the
        # parent still gets the event, and the story carries on -- being stopped
        # mid-sentence for saying "steal" teaches nothing except not to talk.
        # Repeat it enough times in one session and it does stop.
        if safety.action == SafetyAction.REDIRECT:
            strikes = _session_strikes(session) + 1
            timer.add_metadata(safety_strikes=strikes)
            if strikes >= self._strike_limit:
                ended = await self._end_session_for_strikes(
                    request, safety, intent, strikes, timer=timer
                )
                return ended, None
            return None, safety.child_message or _DEFAULT_CAUTION

        return None, None

    async def _resolve_session(self, request: AgentTurnRequest) -> StorySession | None:
        """The session this turn belongs to, even when the caller named none.

        Strikes are counted per session, so a caller that omits `session_id`
        would otherwise get a fresh count every turn and could never be stopped.
        Falling back to the child's newest session closes that: an open one
        keeps accumulating, and an *ended* one is returned as-is so the turn is
        refused. Starting over is then a deliberate act -- POST /v1/sessions,
        the grown-up the cut-off message asks for -- rather than a side effect
        of leaving a field out.
        """
        if request.session_id is not None:
            return await self._repository.get_session(request.session_id)

        latest = await self._repository.latest_session_for_child(request.child_id)
        if latest is not None:
            return latest
        return await self._repository.create_session(
            CreateSessionRequest(
                child_id=request.child_id,
                story_id=request.story_id,
                device_id=request.device_id,
            )
        )

    async def _session_already_ended(
        self,
        request: AgentTurnRequest,
        *,
        timer: LatencyTimer,
    ) -> AgentTurnResponse:
        """Turn away a session that already spent its strikes."""
        timer.add_metadata(session_ended=True)
        return await self._persist_turn(
            request,
            AgentTurnResponse(
                intent=TurnIntent.UNKNOWN,
                response_text=(
                    "We stopped this session earlier. Ask a grown-up to start a new one "
                    "with you and your story will be waiting."
                ),
                safety=SafetyDecision(
                    action=SafetyAction.BLOCK,
                    severity=SafetySeverity.REVIEW,
                    categories=[_SESSION_ENDED],
                    parent_note="A turn arrived after the session ended on the strike limit.",
                ),
            ),
            timer=timer,
        )

    async def _session_history(self, session: StorySession | None) -> list[ConversationTurn]:
        """This session's recent turns, newest first. Conversation state is read from these, never held in memory."""
        if session is None:
            return []
        turns = await self._repository.list_conversation_turns(limit=10)
        return [turn for turn in turns if turn.session_id == session.id]

    async def _conversation_act(
        self,
        request: AgentTurnRequest,
        session: StorySession | None,
        story: Story | None,
        safety: SafetyDecision,
        history: list[ConversationTurn],
        timer: LatencyTimer,
    ) -> AgentTurnResponse | None:
        """Answer what is about the conversation itself (a nod, a repair, a new story) without asking the model."""
        text = request.transcript
        number = len(history)

        def reply(line: str, *notes: TutorNote, intent: TurnIntent = TurnIntent.UNKNOWN, updated: Story | None = None):
            timer.add_metadata(conversation_act=notes[0].kind if notes else "reply")
            return self._persist_turn(
                request,
                AgentTurnResponse(intent=intent, response_text=line, safety=safety, tutor_notes=list(notes), updated_story=updated),
                timer=timer,
            )

        if acts.addresses_parent(text):
            return await reply(acts.GROWN_UP_LINE, TutorNote(kind="parent_address", text="The child spoke to a grown-up."))
        if acts.asks_delete_story(text):
            return await reply(acts.STORY_DELETE_LINE, TutorNote(kind="story_delete_declined", text="The child asked to delete a story."))
        if acts.asks_new_story(text) and request.story_id is not None:
            fresh = await self._repository.create(CreateStoryRequest(title="Untitled Story", synopsis="A new story."))
            if request.session_id is not None:
                await self._repository.attach_session_story(request.session_id, fresh.id)
            return await reply(acts.NEW_STORY_LINE, TutorNote(kind="new_story", text="Started a new story on request."),
                               intent=TurnIntent.EDIT_STORY, updated=fresh)
        if (
            acts.is_frustrated(text)
            and history
            and acts.is_frustrated(history[0].transcript)
            and history[0].response_text != acts.REPAIR_LINE
        ):
            return await reply(
                acts.REPAIR_LINE,
                TutorNote(kind=acts.FRUSTRATION_NOTE, text="The child sounded frustrated twice in a row."),
                TutorNote(kind=acts.DICTATION_NOTE, text="Listening, as the child asked."),
            )
        if acts.wants_to_tell(text):
            return await reply(
                acts.pick(acts.LISTEN_REPLIES, number),
                TutorNote(kind=acts.DICTATION_NOTE, text="The child wants to tell the story themselves."),
            )
        if acts.is_dangling(text) and _consecutive_holds(history) < 2:
            return await reply(acts.pick(acts.HOLD_REPLIES, number), TutorNote(kind=acts.HOLD_LINE_NOTE, text="Waiting for the rest of the sentence."))
        return None

    async def _end_session_for_idle(self, request: AgentTurnRequest, *, timer: LatencyTimer) -> AgentTurnResponse:
        """The same filler line, turn after turn: nobody is talking. Pause instead of answering a loop.

        Uses the ended-session category, so Home Assistant stops listening and a grown-up starts the next session.
        """
        timer.add_metadata(session_ended=True, idle_loop=True)
        response = AgentTurnResponse(
            intent=TurnIntent.UNKNOWN,
            response_text=code_messages.IDLE_PAUSE,
            safety=SafetyDecision(
                action=SafetyAction.BLOCK,
                severity=SafetySeverity.REVIEW,
                categories=[_SESSION_ENDED, _IDLE_LOOP],
                parent_note="The session paused: the same few words were heard several times in a row with nobody speaking.",
            ),
        )
        persisted = await self._persist_turn(request, response, timer=timer)
        if request.session_id is not None:
            await self._repository.end_session(request.session_id)
        return persisted

    async def _end_session_for_strikes(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        intent: TurnIntent,
        strikes: int,
        *,
        timer: LatencyTimer,
    ) -> AgentTurnResponse:
        """Spend the last strike: record it, tell the child calmly, end the session.

        The event is persisted before the session closes so the strike that
        ended it is itself counted and visible to a parent.
        """
        timer.add_metadata(session_ended=True, safety_strikes=strikes)
        decision = safety.model_copy(
            update={
                "severity": SafetySeverity.ALERT,
                "categories": [*safety.categories, "strike_limit_reached"],
                "parent_note": (
                    f"The session ended after {strikes} cautioned turns. "
                    f"Last one: {request.transcript[:200]}"
                ),
            }
        )
        response = AgentTurnResponse(
            intent=intent,
            response_text=(
                "Let's take a break from this story for now. I've left a note for your "
                "grown-up, and we can pick it up again together later."
            ),
            safety=decision,
        )
        persisted = await self._persist_turn(request, response, timer=timer, strike=True)
        if request.session_id is not None:
            with timer.measure("session_end"):
                await self._repository.end_session(request.session_id)
        return persisted

    async def _previous_turn(self, request: AgentTurnRequest) -> ConversationTurn | None:
        """The most recent recorded turn in the same story and session.

        Pending state is derived from this rather than held in memory, so it
        survives a restart and needs no extra state.
        """
        recent = await self._repository.list_conversation_turns(limit=10)
        for turn in recent:
            if turn.story_id != request.story_id:
                continue
            if request.session_id is not None and turn.session_id != request.session_id:
                continue
            return turn
        return None

    async def _ask_where_interest_belongs(
        self,
        response_text: str,
        interest_name: str,
        *,
        request: AgentTurnRequest,
        on_sentence: SentenceSink | None,
    ) -> str:
        """Add the clarifying question, and make sure it is actually heard.

        In the streaming path the reply has already been spoken by the time the
        interest is known -- `interest_candidate` arrives after `response_text`,
        which is ordered first on purpose so audio can start early. The voice
        loop does not re-speak the final text when sentences were streamed, so
        an appended question would never reach the child. It goes to the speaker
        directly instead, through the same gate `_stream_draft` puts every other
        sentence through, and is appended so the saved reply matches what was
        said.
        """
        question = _interest_question(interest_name)
        if question in response_text:
            return response_text

        if on_sentence is None:
            # Nothing has been spoken yet, so the draft's own follow-up can give
            # way to this one and the child is only asked one question.
            return f"{_drop_trailing_question(response_text)} {question}".strip()

        decision = await self._safety.review_assistant_output(
            transcript=request.transcript,
            response_text=question,
            story_patches=[],
        )
        if decision.action == SafetyAction.ALLOW:
            await on_sentence(question)
        return f"{response_text.strip()} {question}".strip()

    async def _record_interest(self, child_id: str, name: str) -> ChildInterest:
        """Save an interest now and look up its description in the background.

        Re-mentioning an interest that already has a description costs nothing:
        the row is returned as-is rather than queued for another lookup.
        """
        existing = await self._repository.get_child_interest(child_id, name)
        if existing is not None and existing.summary:
            return existing

        interest = await self._repository.save_child_interest(
            existing or ChildInterest(child_id=child_id, name=name)
        )
        if self._interest_enricher is not None:
            schedule_enrichment(self._interest_enricher, interest, self._interest_tasks)
        return interest

    async def _resolve_confirmation(
        self,
        request: AgentTurnRequest,
        safety: SafetyDecision,
        story: Story | None,
        awaiting: list[StoryPatch],
        *,
        accepted: bool,
        timer: LatencyTimer,
        interest_name: str | None = None,
    ) -> AgentTurnResponse:
        # The child naming something they love is worth remembering either way.
        # What the yes/no actually settles is whether it also belongs in *this*
        # story, which is what gates the patches below.
        if interest_name is not None:
            with timer.measure("interest_record"):
                await self._record_interest(request.child_id, interest_name)

        if not accepted:
            response = AgentTurnResponse(
                intent=TurnIntent.EDIT_STORY,
                response_text=(
                    f"Got it. I will remember you love {interest_name}. What happens next?"
                    if interest_name is not None
                    else "Okay, I left the story exactly as it was. "
                    "What would you like to do instead?"
                ),
                safety=safety,
                updated_story=story,
                tutor_notes=_interest_tutor_notes(interest_name, in_story=False),
            )
            return await self._persist_turn(request, response, timer=timer)

        applied: list[StoryPatch] = []
        updated_story = story
        with timer.measure("patch_apply"):
            for patch in awaiting:
                # The child has now confirmed, so clear the flag before applying
                # or it would be skipped again and re-offered next turn.
                confirmed = patch.model_copy(update={"child_confirmation_required": False})
                try:
                    updated_story = await self._repository.apply_patch(request.story_id, confirmed)
                except (ValueError, KeyError):
                    # The target is gone (an earlier held change removed it). Skip it rather than failing the turn.
                    logger.warning("dropped a confirmed patch whose target no longer exists: %s", patch.operation.value)
                    continue
                applied.append(confirmed)
        timer.add_metadata(confirmed_patch_count=len(applied))

        if interest_name is not None and not applied:
            # Nothing was staged to apply, so promising a change would be a lie.
            # Hand the pen back instead.
            response_text = f"Got it, {interest_name} belongs in this story. How does it fit in?"
        elif interest_name is not None:
            response_text = f"Done, {interest_name} is in the story now. What happens next?"
        else:
            response_text = "Done. I made that change to the story. What happens next?"

        response = AgentTurnResponse(
            intent=TurnIntent.EDIT_STORY,
            response_text=response_text,
            safety=safety,
            story_patches=applied,
            applied_story_patches=applied,
            updated_story=updated_story,
            tutor_notes=_interest_tutor_notes(interest_name, in_story=True),
        )
        return await self._persist_turn(request, response, timer=timer)

    async def _stream_draft(
        self,
        prompt: str,
        request: AgentTurnRequest,
        on_sentence: SentenceSink,
        *,
        timer: LatencyTimer,
        prefix: str | None,
    ) -> dict[str, object]:
        """Stream the draft, releasing each sentence once it passes a safety gate.

        The gate reviews sentence text only -- the story patches have not been
        generated yet. That is deliberate and does not weaken the guarantee: the
        full review over text *and* patches still runs in `process` before
        anything is applied or persisted. This gate protects what gets spoken,
        the full review protects what gets kept.
        """
        accumulator = SentenceAccumulator()
        payload: dict[str, object] | None = None
        released = 0
        vetoed = False
        # Everything before this is setup; everything after is the child
        # listening. Split out because "first audio" measured from the turn's
        # start hides which half to work on.
        started = time.perf_counter()
        first_text_at: float | None = None
        first_sentence_at: float | None = None

        if prefix:
            await on_sentence(prefix)
            released += 1

        async def release(sentence: str) -> None:
            nonlocal released, vetoed
            if vetoed or not sentence:
                return
            decision = await self._safety.review_assistant_output(
                transcript=request.transcript,
                response_text=sentence,
                story_patches=[],
            )
            if decision.action != SafetyAction.ALLOW:
                # Stop speaking. `process` still runs the full review, and its
                # decision is what the child ultimately hears and what is saved.
                vetoed = True
                timer.add_metadata(sentence_gate_vetoed=True)
                return
            if released == (1 if prefix else 0):
                timer.add_metadata(
                    model_to_first_audio_ms=round((time.perf_counter() - started) * 1000, 1)
                )
            await on_sentence(sentence)
            released += 1

        async for event in self._ollama.stream_json(prompt, model=request.model):
            if event.text:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                    timer.add_metadata(
                        model_to_first_token_ms=round((first_text_at - started) * 1000, 1)
                    )
                for sentence in accumulator.feed(event.text):
                    if first_sentence_at is None:
                        # Written, but not yet gated. The gap between this and
                        # model_to_first_audio_ms is the per-sentence review.
                        first_sentence_at = time.perf_counter()
                        timer.add_metadata(
                            model_to_first_sentence_ms=round(
                                (first_sentence_at - started) * 1000, 1
                            )
                        )
                    await release(sentence)
            if event.field_done:
                # response_text is closed. Speak the tail now rather than
                # waiting for the story patches to finish generating.
                await release(accumulator.flush())
                if released:
                    timer.add_metadata(first_audio_ready=True)
            if event.payload is not None:
                payload = event.payload
                if event.stats:
                    # prompt_eval is time spent reading the prompt before any
                    # token exists. Streaming cannot reduce it, so it is the
                    # floor on first audio and worth seeing in the stage table.
                    timer.add_metadata(
                        prompt_tokens=event.stats.get("prompt_eval_count"),
                        prompt_eval_ms=round(event.stats.get("prompt_eval_duration", 0) / 1e6, 1),
                        generated_tokens=event.stats.get("eval_count"),
                    )

        await release(accumulator.flush())
        timer.add_metadata(streamed_sentence_count=released)
        if payload is None:
            raise OllamaUnavailableError("Ollama stream ended without a JSON object.")
        return payload

    async def _persist_turn(
        self,
        request: AgentTurnRequest,
        response: AgentTurnResponse,
        timer: LatencyTimer | None = None,
        strike: bool = False,
    ) -> AgentTurnResponse:
        if timer is not None:
            timer.add_metadata(
                turn_id=response.turn_id,
                safety_action=response.safety.action.value,
                safety_severity=response.safety.severity.value,
            )
        # Stamped in one place so every path -- including the early returns --
        # tells the caller which session to send the next turn to.
        response = response.model_copy(update={"session_id": request.session_id})
        # Contact details were removed from this turn: the child hears the not-kept line in front of the reply.
        lead = _PII_LEAD.get()
        if lead and lead not in response.response_text:
            response = response.model_copy(update={"response_text": f"{lead} {response.response_text}".strip()})
        turn = ConversationTurn(
            id=response.turn_id,
            child_id=request.child_id,
            story_id=request.story_id,
            session_id=request.session_id,
            device_id=request.device_id,
            intent=response.intent,
            transcript=request.transcript,
            response_text=response.response_text,
            safety=response.safety,
            story_patches=response.story_patches,
            applied_story_patches=response.applied_story_patches,
            tutor_notes=response.tutor_notes,
        )
        if timer is None:
            await self._repository.save_conversation_turn(turn)
        else:
            with timer.measure("save_turn"):
                await self._repository.save_conversation_turn(turn)

        # Every path ends here, which is what makes this the right place to
        # queue the deferred review: no branch can forget to. Only turns the
        # keyword guard passed are worth reviewing -- it has already decided the
        # rest -- and a parent-released line is not re-judged.
        if (
            response.safety.action == SafetyAction.ALLOW
            and "parent_released" not in response.safety.categories
        ):
            self._schedule_input_review(request, turn.id, response.intent)
        if response.safety.action != SafetyAction.ALLOW or response.safety.severity != SafetySeverity.NONE:
            event = SafetyEvent(
                child_id=request.child_id,
                story_id=request.story_id,
                session_id=request.session_id,
                conversation_turn_id=turn.id,
                safety=response.safety,
                transcript_excerpt=request.transcript[:500],
                # A block is a hard stop on a child's turn. It raises a parent
                # alert just as alert_parent does -- it used to be recorded
                # silently, which is how a story about a dog got stopped with
                # nobody told why.
                parent_alert_sent=(
                    response.safety.action in _STOP_ACTIONS
                    and _SESSION_ENDED not in response.safety.categories
                ),
                counted_as_strike=strike,
            )
            if timer is None:
                await self._repository.save_safety_event(event)
            else:
                with timer.measure("save_safety_event"):
                    await self._repository.save_safety_event(event)

        if timer is None:
            memory_chunk = self._memory_builder.build_turn_chunk(turn)
        else:
            with timer.measure("memory_chunk_build"):
                memory_chunk = self._memory_builder.build_turn_chunk(turn)
        if memory_chunk is None:
            return response

        if timer is None:
            memory_chunk.embedding = await embed_document(self._embedder, memory_chunk.content)
            saved_chunks = await self._repository.save_memory_chunks([memory_chunk])
        else:
            with timer.measure("memory_chunk_embedding"):
                memory_chunk.embedding = await embed_document(self._embedder, memory_chunk.content)
            with timer.measure("save_memory_chunks"):
                saved_chunks = await self._repository.save_memory_chunks([memory_chunk])
            timer.add_metadata(memory_chunk_count=len(saved_chunks))
        return response.model_copy(
            update={"memory_chunk_ids": [chunk.id for chunk in saved_chunks]},
        )


def _fallback_response(intent: TurnIntent) -> str:
    match intent:
        case TurnIntent.ASK_DEFINITION:
            return "Nice word choice. Let's define it together, then try it in your story."
        case TurnIntent.ASK_GRAMMAR:
            return "That idea already has a spark. I can help make the sentence smoother while keeping it yours."
        case TurnIntent.CREATE_IMAGE:
            return "That could be a wonderful story picture. I can save it for review, then we can keep writing."
        case TurnIntent.EDIT_STORY:
            return "That edit sounds interesting. Tell me the exact change, and I will keep your story voice intact."
        case TurnIntent.UNDO_LAST_CHANGE:
            return "No problem. I can undo the last story change and help you try a new path."
        case TurnIntent.UNKNOWN:
            return "I did not quite catch how that changes the story. Tell me the part you want to add, change, or explore."
        case _:
            return "I like that story energy. What should the next beat reveal?"


def _picked_story_response(
    intent: TurnIntent,
    safety: SafetyDecision,
    story: Story,
) -> AgentTurnResponse:
    return AgentTurnResponse(
        intent=intent,
        response_text=f"I picked {story.title}. What part should we continue?",
        safety=safety,
        updated_story=story,
        tutor_notes=[
            TutorNote(
                kind="story_selection",
                text="The child selected a saved story by title.",
            )
        ],
    )


def _build_tutor_notes(transcript: str, intent: TurnIntent) -> list[TutorNote]:
    if intent == TurnIntent.ASK_DEFINITION:
        return [TutorNote(kind="definition", text="Definition requests should be answered briefly.")]
    if intent == TurnIntent.ASK_GRAMMAR:
        return [TutorNote(kind="grammar", text="Grammar suggestions should preserve the child's voice.")]
    return []


def _with_confirmation_prompt(
    response_text: str,
    editing_result: StoryEditingPolicyResult,
) -> str:
    if not editing_result.needs_confirmation:
        return response_text
    prompt = "That's a big change. Should I make it?"
    if prompt in response_text:
        return response_text
    return f"{response_text} {prompt}"


# Deliberately short and exact-match only. A false positive here applies a
# destructive edit the child did not ask for, which is the failure this whole
# mechanism exists to prevent, so anything ambiguous falls through to normal
# routing instead of being treated as an answer.
_CONFIRM_YES = frozenset(
    {
        "yes", "yeah", "yep", "yes please", "yes do it", "do it", "go ahead",
        "please do", "sure", "okay do it", "ok do it", "confirm", "make the change",
    }
)
_NO_TAIL = frozenset({"keep", "leave", "it", "that", "thanks", "thank", "you", "dont", "do", "not", "please", "alone", "stop", "as", "is"})
_YES_TAIL = frozenset({"please", "do", "it", "that", "go", "ahead", "thanks", "thank", "you", "make", "the", "change", "remove", "delete"})
_CONFIRM_NO = frozenset(
    {
        "no", "nope", "no thanks", "no thank you", "dont", "do not", "leave it",
        "never mind", "nevermind", "cancel", "stop", "keep it", "dont do it",
        "do not do it", "no dont", "leave it alone",
    }
)


# Recorded on the turn that asks the clarifying question, and read back on the
# next turn to find out which interest the child is answering about.
_INTEREST_CLARIFICATION = "interest_clarification"


def _patches_awaiting_confirmation(previous: ConversationTurn | None) -> list[StoryPatch]:
    """Patches the previous turn proposed and deliberately did not apply."""
    if previous is None:
        return []
    applied = {patch.model_dump_json() for patch in previous.applied_story_patches}
    return [
        patch
        for patch in previous.story_patches
        if patch.child_confirmation_required and patch.model_dump_json() not in applied
    ]


def _interest_awaiting_clarification(previous: ConversationTurn | None) -> str | None:
    """The interest the previous turn asked the child to place, if it asked."""
    if previous is None:
        return None
    for note in previous.tutor_notes:
        if note.kind == _INTEREST_CLARIFICATION and note.text.strip():
            return note.text.strip()
    return None


def _interest_tutor_notes(interest_name: str | None, *, in_story: bool) -> list[TutorNote]:
    if interest_name is None:
        return []
    placement = "part of this story" if in_story else "an outside interest"
    return [
        TutorNote(
            kind="interest",
            text=f"The child confirmed {interest_name} is {placement}.",
        )
    ]


_DEFAULT_CAUTION = "Let's keep that part of the story safe and kind."

# The two actions that stop a turn outright, as opposed to cautioning it.
_STOP_ACTIONS = frozenset({SafetyAction.BLOCK, SafetyAction.ALERT_PARENT})

# Turning a child away because their session is over is an administrative
# refusal, not a finding about anything they said. It is recorded, but it must
# not raise a parent alert, or every word spoken after a cut-off would.
_SESSION_ENDED = "session_ended"
_IDLE_LOOP = "idle_loop"
# What speech recognition invents out of silence or room noise. One of these is just a child being polite;
# the same thing three turns running is a microphone hearing the speaker, so the session pauses.
_FILLER = frozenset({"thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching", "you", "bye", "bye bye"})
IDLE_FILLER_LIMIT = 3


def _transcript_was_released(events: list[SafetyEvent], request: AgentTurnRequest) -> bool:
    """Did a parent already let this exact line through, in this session?

    Scoped to the session on purpose. A parent releasing "she deflates the toys
    sometimes when she bites them" is saying it is fine *now*, in the story they
    just looked at -- not signing it off forever.
    """
    if request.session_id is None:
        return False
    said = request.transcript.strip()
    return any(
        event.session_id == request.session_id
        and event.released_at is not None
        and event.transcript_excerpt.strip() == said
        for event in events
    )


def _pending_recap(events: list[SafetyEvent], story_id: str | None) -> SafetyEvent | None:
    """A released turn that ran while the child was not listening."""
    if story_id is None:
        return None
    for event in events:
        if (
            event.story_id == story_id
            and event.released_at is not None
            and event.recap_delivered_at is None
        ):
            return event
    return None


def _recap_message(transcript: str) -> str:
    return f'Last time we added: "{transcript.strip()}" That is where we left off.'


def _safety_stop_message(decision: SafetyDecision) -> str:
    """What the child hears when a turn is stopped on safety grounds.

    Fixed wording rather than whatever the reviewer improvised. This is the most
    serious thing Yarnotron says, and a child should always learn the same three
    things from it: that it stopped, that this was a safety decision and not a
    glitch, and that a grown-up knows. The model's own reasoning is kept in
    parent_note for the dashboard, where it belongs.

    The prior wording -- "I can't help with that story. Let's try a different
    adventure!" -- said none of that, so a false positive on a dog biting a pool
    toy read as the story simply being rejected.
    """
    if decision.action == SafetyAction.ALERT_PARENT:
        return (
            "I've stopped there and told a grown-up, because this is something "
            "they should know about. Please go and talk with them."
        )
    return (
        "That set off a safety alert, so I stopped there and told a grown-up. "
        "Tell me a different idea and we can keep going."
    )


def _is_worried_child(decision: SafetyDecision) -> bool:
    """Self-harm or a child disclosing something unsafe: the fixed script, never the model's words."""
    text = " ".join(decision.categories).casefold()
    return any(marker in text for marker in ("self_harm", "self-harm", "suicide", "worried_child", "grooming", "abuse"))


def _session_strikes(session: StorySession | None) -> int:
    """Cautioned turns already spent in this session."""
    if session is None:
        return 0
    return session.safety_summary.get(STRIKE_KEY, 0)


def _spoken_prefix(
    caution: str | None,
    recap: str | None,
    selection: "_StorySelection",
) -> str | None:
    """What the streaming path says before the model's reply.

    The caution, the recap and any story intro all have to be spoken up front,
    because the voice loop never re-speaks the final text once sentences have
    streamed.
    """
    parts = [part for part in (caution, _story_intro(selection), recap) if part]
    return " ".join(parts) if parts else None


def _with_caution(response_text: str, caution: str | None) -> str:
    if caution is None or caution in response_text:
        return response_text
    return f"{caution} {response_text}".strip()


def _with_recap(response_text: str, recap: str | None) -> str:
    if recap is None or recap in response_text:
        return response_text
    return f"{recap} {response_text}".strip()


def _interest_question(interest_name: str) -> str:
    return f"Is {interest_name} part of this story, or just something you love?"


# No earlier sentence means the whole reply was the question, and the clarifying question replaces it outright.
_drop_trailing_question = drop_trailing_question


def _kinds(turn: ConversationTurn) -> set[str]:
    return {note.kind for note in turn.tutor_notes}


def _held_fragment(history: list[ConversationTurn], max_age_seconds: int = 120) -> str | None:
    """The unfinished sentence from the last turn, if that turn was a nod waiting for the rest."""
    if not history or acts.HOLD_LINE_NOTE not in _kinds(history[0]):
        return None
    age = (datetime.now(UTC) - history[0].created_at).total_seconds()
    return history[0].transcript if age <= max_age_seconds else None


def _consecutive_holds(history: list[ConversationTurn]) -> int:
    count = 0
    for turn in history:
        if acts.HOLD_LINE_NOTE not in _kinds(turn):
            break
        count += 1
    return count


def _in_dictation(turns: list[ConversationTurn], transcript: str, limit: int = 5) -> bool:
    """True while the child is telling the story: replies are a short nod. After five nods it asks something again."""
    if acts.leaves_dictation(transcript):
        return False
    count = 0
    for turn in turns:
        if acts.DICTATION_NOTE not in _kinds(turn):
            break
        count += 1
    return 0 < count < limit


def _recent_questions(turns: list[ConversationTurn], keep: int = 3) -> list[str]:
    """The closing question of each of the last few replies, newest first, so the Talker does not ask it again."""
    questions: list[str] = []
    for turn in turns:
        sentences = split_sentences(turn.response_text)
        if sentences and sentences[-1].endswith("?") and sentences[-1] not in questions:
            questions.append(sentences[-1])
        if len(questions) >= keep:
            break
    return questions


def _is_filler(transcript: str) -> bool:
    words = "".join(c if c.isalpha() or c == " " else (" " if c == "-" else "") for c in transcript.lower()).split()
    return not words or " ".join(words) in _FILLER


def _filler_run(recent: list[ConversationTurn], session_id: str) -> int:
    """How many of this session's latest turns (newest first in `recent`) were filler, without a gap."""
    run = 0
    for turn in recent:
        if turn.session_id != session_id:
            continue
        if not _is_filler(turn.transcript):
            break
        run += 1
    return run


def _leads_with_yes(transcript: str) -> bool:
    words = transcript.casefold().replace(",", " ").replace(".", " ").replace("!", " ").split()
    return bool(words) and words[0] in {"yes", "yeah", "yep", "yup", "sure"} and len(words) > 1


def _confirmation_answer(transcript: str) -> bool | None:
    """True for yes, False for no, None if this is not an answer at all."""
    normalized = " ".join(
        transcript.casefold().replace("'", "").replace("!", " ").replace(".", " ")
        .replace(",", " ").replace("?", " ").split()
    )
    if normalized in _CONFIRM_YES:
        return True
    if normalized in _CONFIRM_NO:
        return False
    words = normalized.split()
    # "No, keep it" and "yes please do it" are answers. Anything with story words in it is not.
    if words and words[0] in {"no", "nope", "nah"} and all(w in _NO_TAIL for w in words[1:]) and len(words) <= 5:
        return False
    if words and words[0] in {"yes", "yeah", "yep", "yup", "ok", "okay", "sure"} and all(w in _YES_TAIL for w in words[1:]) and len(words) <= 5:
        return True
    return None


def _validate_draft(payload: dict[str, object], timer: LatencyTimer | None = None) -> AgentDraft:
    """Parse a draft, discarding individual items the model got wrong.

    Validating the whole object at once means one hallucinated patch operation
    throws away the entire turn, including a reply that was fine -- and in the
    streaming path that reply has already been spoken, so the child hears it and
    then gets an error. A bad patch should cost only that patch.
    """
    patches = []
    dropped = 0
    for item in payload.get("story_patches") or []:
        try:
            patches.append(StoryPatch.model_validate(item))
        except ValidationError:
            dropped += 1

    notes = []
    for item in payload.get("tutor_notes") or []:
        try:
            notes.append(TutorNote.model_validate(item))
        except ValidationError:
            dropped += 1

    candidate = None
    raw_candidate = payload.get("interest_candidate")
    if raw_candidate:
        try:
            candidate = InterestCandidate.model_validate(raw_candidate)
        except ValidationError:
            dropped += 1

    if dropped and timer is not None:
        timer.add_metadata(dropped_invalid_draft_items=dropped)

    return AgentDraft(
        response_text=str(payload.get("response_text") or ""),
        story_patches=patches,
        tutor_notes=notes,
        interest_candidate=candidate,
    )


# These are spoken aloud, so every word is time the child spends listening
# instead of telling the story. The confirmation prompt used to be 107
# characters -- five seconds, as long as a whole reply should be.
def _new_story_intro(title: str) -> str:
    return f"New story: {title}. Remember that title so we can find it again."


def _resumed_story_intro(title: str) -> str:
    return f"Back to {title}."


def _story_intro(selection: "_StorySelection") -> str | None:
    """The intro `process` will prepend, known before generation starts.

    Streaming speaks it first rather than prepending it afterwards, so both
    paths must produce the same string -- hence the shared builders above.
    """
    if selection.story is None:
        return None
    if selection.created:
        return _new_story_intro(selection.story.title)
    if selection.resumed:
        return _resumed_story_intro(selection.story.title)
    return None


def _with_new_story_intro(response_text: str, story: Story | None) -> str:
    if story is None:
        return response_text
    intro = _new_story_intro(story.title)
    if intro in response_text:
        return response_text
    return f"{intro} {response_text}"


def _with_resumed_story_intro(response_text: str, story: Story | None) -> str:
    if story is None:
        return response_text
    intro = _resumed_story_intro(story.title)
    if intro in response_text:
        return response_text
    return f"{intro} {response_text}"


def _short_acknowledgement(transcript: str) -> str:
    cleaned = " ".join(transcript.strip().split())
    if len(cleaned) <= 90:
        return cleaned
    return f"{cleaned[:87].rstrip()}..."


class _StorySelection(BaseModel):
    story: Story | None = None
    created: bool = False
    resumed: bool = False
    picked: bool = False
    needs_clarification: bool = False
    selection: str = "none"


async def _find_referenced_story(repository: StoryRepository, transcript: str) -> Story | None:
    normalized_transcript = _normalize_story_reference(transcript)
    if not normalized_transcript:
        return None

    stories = await repository.list()
    exact_matches = [
        story for story in stories if _normalize_story_reference(story.title) == normalized_transcript
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    matches = [story for story in stories if _story_title_is_referenced(story, normalized_transcript)]
    if len(matches) == 1 and (
        _asks_for_existing_story(transcript) or _looks_like_story_picker_answer(transcript)
    ):
        return matches[0]
    return None


def _asks_for_story_list(transcript: str) -> bool:
    normalized = _normalize_story_reference(transcript)
    if not normalized:
        return False
    has_story_word = "story" in normalized or "stories" in normalized
    return has_story_word and any(
        phrase in normalized
        for phrase in (
            "what stories",
            "which stories",
            "list stories",
            "show stories",
            "available stories",
            "stories available",
            "saved stories",
            "story list",
        )
    )


def _story_title_is_referenced(story: Story, normalized_transcript: str) -> bool:
    normalized_title = _normalize_story_reference(story.title)
    if not normalized_title:
        return False
    return normalized_title in normalized_transcript


def _story_matches_picker_response(story: Story, transcript: str) -> bool:
    return _normalize_story_reference(story.title) == _normalize_story_reference(transcript)


def _looks_like_story_picker_answer(transcript: str) -> bool:
    normalized = _normalize_story_reference(transcript)
    return bool(normalized) and len(normalized.split()) <= 8


def _sounds_like_ambiguous_continuation(transcript: str) -> bool:
    normalized = _normalize_story_reference(transcript)
    if not normalized:
        return False
    if _asks_for_new_story(transcript):
        return False
    return any(
        phrase in normalized
        for phrase in (
            "continue",
            "keep going",
            "keep writing",
            "where we left off",
            "last story",
            "previous story",
            "old story",
            "same story",
            "our story",
            "the story",
        )
    )


def _asks_for_existing_story(transcript: str) -> bool:
    normalized = _normalize_story_reference(transcript)
    return any(
        phrase in normalized
        for phrase in (
            "continue",
            "resume",
            "open",
            "go back",
            "keep going",
            "where we left off",
            "last story",
            "previous story",
            "old story",
            "same story",
        )
    )


def _asks_for_new_story(transcript: str) -> bool:
    normalized = _normalize_story_reference(transcript)
    return any(
        phrase in normalized
        for phrase in (
            "new story",
            "start a story",
            "start the story",
            "write a story",
            "make a story",
            "create a story",
            "tell a story",
            "story called",
            "story titled",
            "story named",
        )
    )


def _first_turn_story_title(transcript: str) -> str:
    explicit = _extract_explicit_title(transcript)
    if explicit:
        return explicit

    candidate = _normalize_title_candidate(transcript)
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9']+", candidate)
        if word.casefold()
        not in {
            "a",
            "an",
            "the",
            "to",
            "of",
            "and",
            "with",
            "about",
            "story",
            "add",
            "new",
            "start",
            "write",
            "make",
            "create",
            "tell",
        }
    ]
    if not words:
        return "Untitled Story"
    return " ".join(word.capitalize() for word in words[:7])


def _extract_explicit_title(transcript: str) -> str | None:
    patterns = (
        r"(?:story\s+)?(?:called|titled|named)\s+(.+)$",
        r"title\s+(?:is|should be)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, transcript.strip(), flags=re.IGNORECASE)
        if match is None:
            continue
        title = _trim_title_tail(match.group(1))
        if title:
            return title
    return None


def _trim_title_tail(value: str) -> str:
    title = value.strip().strip("\"'.,!?")
    for separator in (" about ", " where ", " with ", " and then ", ". "):
        index = title.casefold().find(separator)
        if index > 0:
            title = title[:index]
    return " ".join(title.split())[:120].strip()


def _normalize_title_candidate(transcript: str) -> str:
    candidate = transcript.strip()
    candidate = re.sub(
        r"^(?:let'?s|lets|please|can we|could we|i want to|we should)\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(
        r"^(?:start|write|make|create|tell)\s+(?:a\s+|the\s+)?(?:new\s+)?story(?:\s+about)?\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    return candidate


def _normalize_story_reference(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def get_turn_processor(
    ollama: OllamaClient = Depends(get_ollama_client),
    repository: StoryRepository = Depends(get_story_repository),
    embedder: Embedder = Depends(get_embedder),
    memory_builder: StoryMemoryBuilder = Depends(get_memory_builder),
    safety: SafetyReviewer = Depends(get_safety_reviewer),
    settings: Settings = Depends(get_settings),
) -> TurnProcessor:
    engine = (
        AgentEngine(llm=ollama, safety=safety, scribe_budget_seconds=settings.agent_scribe_budget_seconds)
        if settings.engine == "agent"
        else None
    )
    return TurnProcessor(
        engine=engine,
        ollama=ollama,
        repository=repository,
        embedder=embedder,
        memory_builder=memory_builder,
        safety=safety,
        strike_limit=settings.safety_strike_limit,
        interest_enricher=InterestEnricher(
            ollama=ollama,
            repository=repository,
            safety=safety,
        ),
    )
