"""Build a short description of something a child is into.

Runs off the turn path. A child who says "make it wobble like in Kerbal Space
Program" should not wait on a lookup before hearing a reply, so the interest is
recorded immediately with no summary and filled in a moment later. The prompt
already tolerates a summary-less interest -- the name alone is usually enough
for the model to work with, and the summary just sharpens it from the next turn.

The description comes from the model already running locally rather than from a
web search. Nothing about a child leaves the machine, which is the tradeoff this
project makes everywhere else too. The cost is that the model will not know
every niche thing a child loves, so the prompt gives it an explicit way to say
so and an unknown interest simply keeps its name.
"""

from __future__ import annotations

import asyncio
import logging

from yarnotron.agent.safety import SafetyReviewer
from yarnotron.models.interests import ChildInterest
from yarnotron.models.safety import SafetyAction
from yarnotron.services.ollama import OllamaClient, OllamaUnavailableError
from yarnotron.services.story_store import StoryRepository

logger = logging.getLogger(__name__)

# Anything longer stops being a hint and starts crowding the story out of the
# prompt, which is the thing agent/story_context exists to prevent.
MAX_SUMMARY_CHARS = 300

UNKNOWN = "UNKNOWN"


def build_interest_prompt(name: str) -> str:
    return f"""Describe "{name}" for a storyteller working with a 12-year-old.

Two or three sentences, at most {MAX_SUMMARY_CHARS} characters. Cover what it is
and the details that would be fun to borrow in a story: the mood, the look, the
kinds of things that happen in it.

If you are not confident you know what "{name}" is, reply with exactly {UNKNOWN}
and nothing else. A wrong description is worse than none, because the child will
notice and it will not sound like the thing they love.

Write the description only, with no preamble and no quotation marks.
"""


class InterestEnricher:
    def __init__(
        self,
        ollama: OllamaClient,
        repository: StoryRepository,
        safety: SafetyReviewer,
    ) -> None:
        self._ollama = ollama
        self._repository = repository
        self._safety = safety

    async def enrich(self, interest: ChildInterest) -> ChildInterest:
        """Look the interest up and save the summary. Returns the saved row."""
        summary = await self._summarize(interest.name)
        if summary is None:
            return interest
        return await self._repository.save_child_interest(
            interest.model_copy(update={"summary": summary})
        )

    async def _summarize(self, name: str) -> str | None:
        try:
            raw = await self._ollama.generate(build_interest_prompt(name))
        except OllamaUnavailableError:
            # The turn itself already succeeded. An interest with no summary is
            # the documented resting state, so there is nothing to recover.
            logger.warning("interest lookup unavailable for %r", name)
            return None

        summary = raw.strip().strip('"')
        if not summary or summary.upper().startswith(UNKNOWN):
            return None
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[: MAX_SUMMARY_CHARS - 1].rstrip() + "…"

        # This text is written by a model and lands in a child's prompt, so it
        # goes through the same gate as anything else the model produces.
        decision = await self._safety.review_assistant_output(
            transcript=name,
            response_text=summary,
            story_patches=[],
        )
        if decision.action != SafetyAction.ALLOW:
            logger.warning("interest summary blocked by safety review for %r", name)
            return None
        return summary


def schedule_enrichment(
    enricher: InterestEnricher,
    interest: ChildInterest,
    tasks: set[asyncio.Task],
) -> asyncio.Task:
    """Fire the lookup without awaiting it.

    `tasks` holds a strong reference until the task finishes; asyncio only keeps
    a weak one, so a task that nothing else holds can be garbage collected
    mid-flight.
    """

    async def run() -> None:
        try:
            await enricher.enrich(interest)
        except Exception:
            # Never let a background lookup surface as a failed story turn.
            logger.exception("interest enrichment failed for %r", interest.name)

    task = asyncio.create_task(run())
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task
