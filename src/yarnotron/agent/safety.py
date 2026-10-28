from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from fastapi import Depends
from pydantic import ValidationError

from yarnotron.core.config import Settings, get_settings
from yarnotron.models.safety import SafetyAction, SafetyDecision, SafetySeverity
from yarnotron.models.story import Story, StoryPatch
from yarnotron.services.ollama import (
    OllamaClient,
    OllamaMalformedJsonError,
    get_ollama_client,
)

logger = logging.getLogger(__name__)


def _contains_term(text: str, terms: set[str]) -> bool:
    """Whole words only.

    These were substring matches, which is how "whatever" was flagged for hate,
    "stealth" for theft, and -- at the top tier, which alerts a parent and stops
    the turn -- "unisex" for sexual content. A term list this short cannot afford
    to fire on the middle of ordinary words.
    """
    return any(re.search(rf"\b{re.escape(term)}(?:s|es|ed|d|ing)?\b", text) for term in terms)


class SafetyClassifier:
    """First-pass local safety classifier.

    This is intentionally conservative and simple. We will replace or augment it
    with a local classifier model before any child-facing use.
    """

    _alert_terms = {
        "suicide",
        "kill myself",
        "self harm",
        "nude",
        "sex",
        "porn",
        "explicit",
    }

    _self_harm_terms = {"suicide", "kill myself", "self harm"}

    # "hate" is deliberately absent. Children say "I hate that name" and "I hate
    # broccoli" constantly, and as a bare word it carried almost no signal about
    # hate speech while costing a caution and a strike every time. The model
    # reviewer still weighs hate and harassment, with the sentence around it --
    # which is what tells contempt for real people from a dislike of a character.
    _redirect_terms = {
        "gore",
        "torture",
        "graphic violence",
        "drug",
        "steal",
        "bully",
    }

    # A child telling us something is wrong. First person only, so a story character saying these things is
    # not caught. The words are the child's cue for the fixed, approved script and a parent alert.
    _worried_patterns = tuple(
        re.compile(pattern)
        for pattern in (
            r"\b(told|tells|asked|asks|made|makes) me (to |not to |never to )?(not |never )?(tell|say)\b",
            r"\bour (little )?secret\b(?! (hideout|base|club|clubhouse|cave|treasure|plan|door|garden|place|room))",
            r"\b(touched|touches|touching) me\b",
            r"\bsomeone (is |keeps )?(hurting|touching|hitting|scaring) me\b",
            r"\bmy (uncle|aunt|dad|mom|mum|stepdad|stepmom|coach|teacher|babysitter|neighbou?r|brother|sister|cousin)"
            r"[^.?!]{0,30}\b(hurts?|hits?|touch\w*|scares?) me\b",
            r"\bi (am|m|feel) (not safe|unsafe|scared to go home|afraid of (him|her|them|my))\b",
            r"\bi (dont|do not|don't) feel safe\b",
        )
    )

    def classify(self, transcript: str) -> SafetyDecision:
        normalized = transcript.lower().replace("\u2019", "'")
        # Letters said one at a time ("S E X", "a n u d e") are read as the word they spell.
        for run in re.findall(r"\b(?:[a-z]\s+){2,}[a-z]\b", normalized):
            spelled = re.sub(r"\s+", "", run)
            normalized += "".join(f" {term}" for term in self._alert_terms | self._redirect_terms if term in spelled)

        if any(pattern.search(normalized) for pattern in self._worried_patterns):
            return SafetyDecision(
                action=SafetyAction.ALERT_PARENT,
                severity=SafetySeverity.ALERT,
                categories=["child_safety_review", "worried_child"],
                child_message=(
                    "Thank you for telling me. That sounds like something a grown-up should know about."
                ),
                parent_note="The child said something that may mean they are unsafe or being asked to keep a secret.",
            )

        if _contains_term(normalized, self._alert_terms):
            categories = ["child_safety_review"]
            if _contains_term(normalized, self._self_harm_terms):
                categories.append("self_harm")
            return SafetyDecision(
                action=SafetyAction.ALERT_PARENT,
                severity=SafetySeverity.ALERT,
                categories=categories,
                child_message=(
                    "I can't help make that kind of story. I'm going to save a note "
                    "for your parent because this is something an adult should know about."
                ),
                parent_note="The transcript matched an initial high-severity safety keyword.",
            )

        if _contains_term(normalized, self._redirect_terms):
            return SafetyDecision(
                action=SafetyAction.REDIRECT,
                severity=SafetySeverity.REVIEW,
                categories=["redirect_recommended"],
                child_message=(
                    "Let's steer that in a safer direction. We can keep the drama, "
                    "but make it more adventurous than harmful."
                ),
                parent_note="The transcript matched an initial redirect keyword.",
            )

        return SafetyDecision()


def build_story_safety_context(story: Story | None) -> dict[str, object] | None:
    """Just enough for the reviewer to know who is in the story.

    A single spoken line is often innocent only in context: "she deflates the
    toys sometimes when she bites them" reads very differently once you know
    "she" is a Newfypoo. Reviewed bare, gemma4:12b blocked that as sexual
    content three times out of three; with this attached it allowed it three
    times out of three.

    Deliberately *not* the scenes, the draft text, or the child's earlier
    lines. Those are words a child writes, and feeding them to the reviewer
    would let someone spend a few turns establishing framing that talks it into
    allowing the next thing said. Names and roles are enough to disambiguate a
    dog from a person, and are a poor lever for anything else.
    """
    if story is None:
        return None
    return {
        "title": story.title,
        "characters": [
            {key: value for key, value in (("name", c.name), ("role", c.role)) if value}
            for c in story.characters
        ],
    }


class SafetyReviewer(Protocol):
    def classify_transcript(self, transcript: str) -> SafetyDecision:
        """The instant keyword verdict, with no model call behind it."""
        ...

    async def review_transcript(
        self,
        transcript: str,
        story: Story | None = None,
        intent: str | None = None,
    ) -> SafetyDecision: ...

    async def review_assistant_output(
        self,
        transcript: str,
        response_text: str,
        story_patches: list[StoryPatch],
    ) -> SafetyDecision: ...


class ModelSafetyReviewer:
    def __init__(
        self,
        ollama: OllamaClient,
        settings: Settings,
        keyword_classifier: SafetyClassifier | None = None,
    ) -> None:
        self._ollama = ollama
        self._settings = settings
        self._keyword_classifier = keyword_classifier or SafetyClassifier()

    def classify_transcript(self, transcript: str) -> SafetyDecision:
        return self._keyword_classifier.classify(transcript)

    async def review_transcript(
        self,
        transcript: str,
        story: Story | None = None,
        intent: str | None = None,
    ) -> SafetyDecision:
        keyword_decision = self._keyword_classifier.classify(transcript)
        if keyword_decision.action == SafetyAction.ALERT_PARENT:
            return keyword_decision

        payload: dict[str, object] = {"transcript": transcript}
        story_context = build_story_safety_context(story)
        if story_context is not None:
            payload["story"] = story_context
        if intent is not None:
            payload["intent"] = intent

        model_decision = await self._review(
            stage="child_transcript",
            payload=payload,
        )
        return combine_safety_decisions(keyword_decision, model_decision)

    async def review_assistant_output(
        self,
        transcript: str,
        response_text: str,
        story_patches: list[StoryPatch],
    ) -> SafetyDecision:
        return await self._review(
            stage="assistant_output",
            payload={
                "child_transcript": transcript,
                "assistant_response": response_text,
                "story_patches": [patch.model_dump(mode="json") for patch in story_patches],
            },
        )

    async def _review(self, stage: str, payload: dict[str, object]) -> SafetyDecision:
        """Ask the reviewer, once more if the answer was unreadable.

        A garbled transcript can make a small model emit broken JSON -- phase 0
        measured about 3% of calls. That used to become a redirect, so the child
        was cautioned, and later charged a strike, for a model glitch. It is not
        evidence of anything, so a second attempt is made and then the turn is
        allowed with an event left for the parent. The keyword classifier, the
        output review and the per-sentence gate all still apply; this is not the
        only thing standing between a child and unsafe content.
        """
        prompt = _build_safety_prompt(stage=stage, payload=payload)
        for attempt in (1, 2):
            try:
                data = await self._ollama.generate_json(
                    prompt,
                    model=self._settings.safety_model,
                    options={"num_predict": 160},
                    think=False,
                )
                return _normalize_model_decision(SafetyDecision.model_validate(data))
            except (ValidationError, OllamaMalformedJsonError):
                # Deliberately not OllamaUnavailableError: Ollama being down is a
                # real failure and must reach the caller, not be waved through.
                logger.warning(
                    "safety reviewer returned unreadable output at stage %s (attempt %d)",
                    stage,
                    attempt,
                )
        return SafetyDecision(
            action=SafetyAction.ALLOW,
            severity=SafetySeverity.INFO,
            categories=["safety_model_invalid_output"],
            parent_note=(
                "The safety reviewer returned unreadable output twice, so this turn "
                "was not reviewed by the model. The keyword check and the output "
                "review still ran."
            ),
        )


def combine_safety_decisions(*decisions: SafetyDecision) -> SafetyDecision:
    ranked = sorted(decisions, key=lambda decision: _risk_rank(decision), reverse=True)
    strongest = ranked[0] if ranked else SafetyDecision()
    categories = []
    for decision in decisions:
        for category in decision.categories:
            if category not in categories:
                categories.append(category)
    return strongest.model_copy(update={"categories": categories or strongest.categories})


def get_safety_reviewer(
    ollama: OllamaClient = Depends(get_ollama_client),
    settings: Settings = Depends(get_settings),
) -> SafetyReviewer:
    return ModelSafetyReviewer(ollama=ollama, settings=settings)


def _risk_rank(decision: SafetyDecision) -> int:
    action_rank = {
        SafetyAction.ALLOW: 0,
        SafetyAction.REDIRECT: 2,
        SafetyAction.BLOCK: 3,
        SafetyAction.ALERT_PARENT: 4,
    }[decision.action]
    severity_rank = {
        SafetySeverity.NONE: 0,
        SafetySeverity.INFO: 1,
        SafetySeverity.REVIEW: 2,
        SafetySeverity.ALERT: 3,
        SafetySeverity.URGENT: 4,
    }[decision.severity]
    return max(action_rank, severity_rank)


def _normalize_model_decision(decision: SafetyDecision) -> SafetyDecision:
    if decision.action == SafetyAction.ALLOW and decision.severity != SafetySeverity.NONE:
        return decision.model_copy(update={"action": SafetyAction.REDIRECT})
    if decision.action != SafetyAction.ALLOW and decision.child_message is None:
        return decision.model_copy(update={"child_message": _default_child_message(decision)})
    return decision


def _default_child_message(decision: SafetyDecision) -> str:
    if decision.action == SafetyAction.ALERT_PARENT:
        return (
            "I can't help make that kind of story. I'm going to save a note for your "
            "parent because this is something an adult should know about."
        )
    return "Let's steer that in a safer direction and keep the story adventurous."


def _build_safety_prompt(stage: str, payload: dict[str, object]) -> str:
    return f"""You are a strict child-safety reviewer for a local storytelling app used by a 12-year-old.

Review the content for safety before it is spoken, stored, or applied to a story.

Return only valid JSON matching this schema:
{{
  "action": "allow | redirect | block | alert_parent",
  "severity": "none | info | review | alert | urgent",
  "categories": ["short_category"],
  "child_message": "short gentle message or null",
  "parent_note": "short note for parent dashboard or null"
}}

Safety categories to consider:
- sexual content, nudity, grooming, romantic adult-child themes
- self-harm or suicide
- graphic violence, gore, torture, cruelty
- weapons instructions, illegal acts, drugs
- hate, harassment, bullying
- sharing private personal information
- the child saying someone is hurting them, touching them, scaring them, or asking them to keep a secret (use alert_parent with the category worried_child)
- dangerous dares or unsafe real-world behavior
- generated media requests involving risky content

Guidance:
- A `story` block, when present, is the setting the child is already writing in:
  its title and who is in it. Read the transcript in that context. A character
  whose role says she is a dog is a dog, and a line about her biting a pool toy
  is a line about a dog biting a pool toy. Judging a single sentence with no
  setting is how ordinary play gets mistaken for something else.
- Context explains, it never excuses. Nothing in `story` makes sexual content,
  self-harm or real-world danger acceptable, and a character described as an
  adult, a monster, or a machine does not change what is safe for the child
  writing it.
- An `intent` of edit_story or undo_last_change means the child is editing, and
  is talking to you about their draft rather than about a person. "Get rid of
  the speaker", "remove her", "never mention it again" are instructions about
  the text. Deleting a character is not violence and refusing to mention
  something is not harassment. Judge the story content they are asking for, not
  the tone of the instruction.
- Disliking part of a story is not hate. "I hate that name", "she's annoying",
  "this bit is stupid" are opinions about fiction a child is writing, and are
  allow/none. Hate means contempt for real people over who they are.
- Transcripts arrive from speech-to-text and are often misheard. A line that is
  ungrammatical, cut off, or simply does not mean anything -- "the yellow dwarf
  that is in its made wife" -- is a transcription error, not a coded message.
  Allow it; the child will say it again. Never infer a hidden meaning from words
  that only look odd because they were misheard.
- Use allow/none for age-appropriate adventure, mild fantasy conflict, and ordinary story tension.
- Use redirect/review for concerning story ideas that can be made safe.
- Use block/alert or alert_parent/alert for sexual content, self-harm, credible harm, or urgent adult involvement.
- Keep child_message calm, brief, and non-shaming.

Stage: {stage}
Content:
{json.dumps(payload, indent=2)}
"""
