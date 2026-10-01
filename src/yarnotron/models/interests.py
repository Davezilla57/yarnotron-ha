from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


def interest_key(name: str) -> str:
    """Dedupe key for an interest name.

    A child will say "Kerbal Space Program", "kerbal space program" and
    "Kerbal  Space  Program" across sessions and mean one thing.
    """
    return " ".join(name.casefold().split())


class ChildInterest(BaseModel):
    """Something a child cares about, outside any one story.

    Scoped to the child rather than to a story: the point of the feature is that
    an interest picked up while writing one story is available in the next one.
    """

    id: str = Field(default_factory=lambda: f"interest_{uuid4().hex}")
    child_id: str
    name: str = Field(min_length=1, max_length=80)
    # Filled in by the background lookup, so an interest is usable the moment it
    # is recorded and gets richer a turn later.
    summary: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def key(self) -> str:
        return interest_key(self.name)


class InterestCandidate(BaseModel):
    """An outside interest the model spotted in what the child just said."""

    name: str = Field(min_length=1, max_length=80)
    # True when the model cannot tell whether the mention is story content or
    # just something the child likes. This is what triggers the clarifying
    # question instead of a silent guess.
    ambiguous: bool = False
