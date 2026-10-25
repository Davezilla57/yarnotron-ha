import hashlib
import math
from typing import Protocol

from fastapi import Depends

from yarnotron.core.config import Settings, get_settings
from yarnotron.models.memory import ConversationTurn, MemoryChunk
from yarnotron.models.story import StoryPatch
from yarnotron.services.ollama import OllamaClient, get_ollama_client


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class StoryMemoryBuilder:
    def build_turn_chunk(self, turn: ConversationTurn) -> MemoryChunk | None:
        if turn.story_id is None:
            return None

        content = _build_turn_memory_content(turn)
        return MemoryChunk(
            story_id=turn.story_id,
            conversation_turn_id=turn.id,
            content=content,
            metadata={
                "kind": "conversation_turn",
                "intent": turn.intent.value,
                "safety_action": turn.safety.action.value,
                "safety_severity": turn.safety.severity.value,
                "patch_operations": [patch.operation.value for patch in turn.applied_story_patches],
            },
        )


def _with_prefix(prefix: str, text: str) -> str:
    return f"{prefix.strip()} {text}" if prefix.strip() else text


class OllamaEmbedder:
    """Embeds through whichever model server is configured (Ollama or OpenAI-compatible).

    `embed` is the passage form. Models that distinguish a search query from a
    stored passage get their prefix from the two methods below; with no prefix
    configured all three are identical.
    """

    def __init__(self, settings: Settings, ollama: OllamaClient) -> None:
        self._settings = settings
        self._ollama = ollama

    async def embed(self, text: str) -> list[float]:
        return await self.embed_document(text)

    async def embed_document(self, text: str) -> list[float]:
        return await self._embed(text, self._settings.embedding_document_prefix)

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed(text, self._settings.embedding_query_prefix)

    async def _embed(self, text: str, prefix: str) -> list[float]:
        if self._settings.embedding_model:
            embedding = await self._ollama.embed(
                _with_prefix(prefix, text), self._settings.embedding_model
            )
            return _fit_dimensions(embedding, self._settings.embedding_dimensions)
        return deterministic_embedding(text, self._settings.embedding_dimensions)


async def embed_query(embedder: Embedder, text: str) -> list[float]:
    """Embed `text` as a search query; falls back to plain `embed` for simple embedders."""
    method = getattr(embedder, "embed_query", None)
    return await (method(text) if method is not None else embedder.embed(text))


async def embed_document(embedder: Embedder, text: str) -> list[float]:
    """Embed `text` as a stored passage; falls back to plain `embed` for simple embedders."""
    method = getattr(embedder, "embed_document", None)
    return await (method(text) if method is not None else embedder.embed(text))


def deterministic_embedding(text: str, dimensions: int = 768) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimensions:
                break
        counter += 1

    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0:
        return values
    return [value / magnitude for value in values]


def get_embedder(
    settings: Settings = Depends(get_settings),
    ollama: OllamaClient = Depends(get_ollama_client),
) -> Embedder:
    return OllamaEmbedder(settings=settings, ollama=ollama)


def get_memory_builder() -> StoryMemoryBuilder:
    return StoryMemoryBuilder()


def _build_turn_memory_content(turn: ConversationTurn) -> str:
    applied_changes = _format_patches(turn.applied_story_patches)
    proposed_changes = _format_patches(turn.story_patches)

    parts = [
        f"Child said: {turn.transcript}",
        f"Yarnotron replied: {turn.response_text}",
    ]
    if applied_changes:
        parts.append(f"Applied story changes: {applied_changes}")
    elif proposed_changes:
        parts.append(f"Proposed story changes: {proposed_changes}")
    if turn.tutor_notes:
        parts.append(
            "Tutor notes: "
            + "; ".join(f"{note.kind}: {note.text}" for note in turn.tutor_notes)
        )
    return "\n".join(parts)


def _format_patches(patches: list[StoryPatch]) -> str:
    return "; ".join(
        f"{patch.operation.value} {patch.data}" for patch in patches
    )


def _fit_dimensions(embedding: list[float], dimensions: int) -> list[float]:
    if len(embedding) == dimensions:
        return embedding
    if len(embedding) > dimensions:
        return embedding[:dimensions]
    return embedding + [0.0] * (dimensions - len(embedding))
