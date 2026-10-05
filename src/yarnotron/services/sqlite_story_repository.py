"""The implementation of StoryRepository, so the datastore is just a file.

Phase 3 of the teardown: an app a parent installs by dragging an icon cannot
also require Docker and a database server.

Design notes:

- Each row keeps its pydantic model serialised in a `document` column. Extra
  columns exist only where something is filtered or ordered by them, which
  keeps the schema small.
- Vector search uses sqlite-vec's `vec0` virtual table. It stores no metadata,
  so `memory_chunks` holds the row and `memory_vectors` holds the embedding,
  joined on the chunk id.
- Timestamps are ISO-8601 text. SQLite has no timestamp type, and text sorts
  correctly for the ordering this needs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import sqlite_vec

from yarnotron.core.config import Settings
from yarnotron.models.media import (
    CreateMediaAssetRequest,
    MediaAsset,
    MediaAssetStatus,
    ReviewMediaAssetRequest,
)
from yarnotron.models.interests import ChildInterest, interest_key
from yarnotron.models.memory import ConversationTurn, MemoryChunk, MemorySearchResult
from yarnotron.models.operations import (
    STRIKE_KEY,
    CreateSessionRequest,
    ReleaseSafetyEventRequest,
    RestoreStoryVersionResponse,
    ReviewSafetyEventRequest,
    SafetyEvent,
    StorySession,
    StoryVersion,
)
from yarnotron.models.safety import SafetyDecision
from yarnotron.models.status import RepositorySummary
from yarnotron.models.story import CreateStoryRequest, Story, StoryPatch
from yarnotron.services.story_store import StoryPatchApplier, story_id_from_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS stories (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS stories_updated_at_idx ON stories(updated_at DESC);

CREATE TABLE IF NOT EXISTS story_versions (
    id TEXT PRIMARY KEY,
    story_id TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS story_versions_story_idx
    ON story_versions(story_id, created_at DESC);

CREATE TABLE IF NOT EXISTS story_patches (
    id TEXT PRIMARY KEY,
    story_id TEXT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    data TEXT NOT NULL,
    child_confirmation_required INTEGER NOT NULL DEFAULT 0,
    rationale TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS story_patches_story_idx ON story_patches(story_id);

CREATE TABLE IF NOT EXISTS story_sessions (
    id TEXT PRIMARY KEY,
    child_id TEXT NOT NULL,
    story_id TEXT REFERENCES stories(id) ON DELETE SET NULL,
    document TEXT NOT NULL,
    started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    story_id TEXT REFERENCES stories(id) ON DELETE SET NULL,
    session_id TEXT REFERENCES story_sessions(id) ON DELETE SET NULL,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversation_turns_created_idx
    ON conversation_turns(created_at DESC);

CREATE TABLE IF NOT EXISTS safety_events (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES story_sessions(id) ON DELETE SET NULL,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS safety_events_created_idx ON safety_events(created_at DESC);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id TEXT PRIMARY KEY,
    story_id TEXT REFERENCES stories(id) ON DELETE CASCADE,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_chunks_story_idx ON memory_chunks(story_id);

CREATE TABLE IF NOT EXISTS media_assets (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS media_assets_created_idx ON media_assets(created_at DESC);

-- Interests belong to the child, not to a story, so there is no story_id here.
-- name_key is the casefolded name; the unique index is what makes re-mentioning
-- an interest update it rather than pile up duplicates.
CREATE TABLE IF NOT EXISTS child_interests (
    id TEXT PRIMARY KEY,
    child_id TEXT NOT NULL,
    name_key TEXT NOT NULL,
    document TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS child_interests_child_name_idx
    ON child_interests(child_id, name_key);
"""


def database_path(settings: Settings) -> Path:
    """Resolve the sqlite file from the configured URL.

    Accepts `sqlite:///absolute/path.db`, `sqlite://relative.db` and a bare
    filesystem path, so the setting can be written either way.
    """
    url = settings.database_url
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///") :]).expanduser()
    if url.startswith("sqlite://"):
        return Path(url[len("sqlite://") :]).expanduser()
    return Path(url).expanduser()


class SqliteStoryRepository:
    def __init__(self, settings: Settings, path: Path | None = None) -> None:
        self._settings = settings
        self._path = path or database_path(settings)
        self._dimensions = settings.embedding_dimensions

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as conn:
            await conn.executescript(SCHEMA)
            await conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
                f"USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{self._dimensions}])"
            )
            await conn.commit()

    def _connect(self) -> aiosqlite.Connection:
        connection = aiosqlite.connect(self._path)
        connection.daemon = True
        return _VecConnection(connection)

    async def checkpoint(self) -> str:
        """Fold the write-ahead log into the database file, so the file on disk is whole.

        Harmless when the database is not in WAL mode (SQLite answers with
        zeros). Returns a short description for the shutdown report.
        """
        async with self._connect() as conn:
            cursor = await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            row = await cursor.fetchone()
        busy, log_frames, checkpointed = (row or (0, 0, 0))[:3]
        if busy:
            return f"incomplete: another connection was busy ({checkpointed}/{log_frames} frames)"
        return "ok"

    # ---------------------------------------------------------------- stories

    async def create(self, request: CreateStoryRequest) -> Story:
        async with self._connect() as conn:
            story_id = await _next_story_id(conn, request.title)
            story = Story(id=story_id, title=request.title, synopsis=request.synopsis)
            await conn.execute(
                "INSERT INTO stories (id, document, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (
                    story.id,
                    story.model_dump_json(),
                    _iso(story.created_at),
                    _iso(story.updated_at),
                ),
            )
            await conn.commit()
        return story

    async def list(self) -> list[Story]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT document FROM stories ORDER BY updated_at DESC"
            )
        return [Story.model_validate_json(row[0]) for row in rows]

    async def get(self, story_id: str) -> Story | None:
        async with self._connect() as conn:
            row = await _fetchone(conn, "SELECT document FROM stories WHERE id = ?", (story_id,))
        return Story.model_validate_json(row[0]) if row else None

    async def delete(self, story_id: str) -> bool:
        async with self._connect() as conn:
            cursor = await conn.execute("DELETE FROM stories WHERE id = ?", (story_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def apply_patch(self, story_id: str, patch: StoryPatch) -> Story:
        async with self._connect() as conn:
            row = await _fetchone(conn, "SELECT document FROM stories WHERE id = ?", (story_id,))
            if row is None:
                raise KeyError(story_id)
            current = Story.model_validate_json(row[0])
            now = datetime.now(UTC)

            # Snapshot before mutating, so undo has somewhere to go back to.
            await conn.execute(
                "INSERT INTO story_versions (id, story_id, document, created_at) "
                "VALUES (?, ?, ?, ?)",
                (f"version_{uuid4().hex}", story_id, current.model_dump_json(), _iso(now)),
            )
            await conn.execute(
                "INSERT INTO story_patches "
                "(id, story_id, operation, data, child_confirmation_required, rationale, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"patch_{uuid4().hex}",
                    story_id,
                    patch.operation.value,
                    json.dumps(patch.data),
                    int(patch.child_confirmation_required),
                    patch.rationale,
                    _iso(now),
                ),
            )

            updated = StoryPatchApplier().apply(current, patch)
            updated.updated_at = now
            await conn.execute(
                "UPDATE stories SET document = ?, updated_at = ? WHERE id = ?",
                (updated.model_dump_json(), _iso(now), story_id),
            )
            await conn.commit()
        return updated

    # --------------------------------------------------------------- versions

    async def list_versions(self, story_id: str) -> list[StoryVersion]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, story_id, document, created_at FROM story_versions "
                "WHERE story_id = ? ORDER BY created_at DESC, rowid DESC",
                (story_id,),
            )
        return [
            StoryVersion(
                id=row[0],
                story_id=row[1],
                document=Story.model_validate_json(row[2]),
                created_at=_parse(row[3]),
            )
            for row in rows
        ]

    async def restore_version(self, story_id: str, version_id: str) -> RestoreStoryVersionResponse:
        async with self._connect() as conn:
            current = await _fetchone(
                conn, "SELECT document FROM stories WHERE id = ?", (story_id,)
            )
            if current is None:
                raise KeyError(story_id)
            version = await _fetchone(
                conn,
                "SELECT document FROM story_versions WHERE id = ? AND story_id = ?",
                (version_id, story_id),
            )
            if version is None:
                raise KeyError(version_id)

            now = datetime.now(UTC)
            previous_version_id = f"version_{uuid4().hex}"
            # Snapshot what we are replacing, so an accidental undo is undoable.
            await conn.execute(
                "INSERT INTO story_versions (id, story_id, document, created_at) "
                "VALUES (?, ?, ?, ?)",
                (previous_version_id, story_id, current[0], _iso(now)),
            )
            restored = Story.model_validate_json(version[0])
            restored.updated_at = now
            await conn.execute(
                "UPDATE stories SET document = ?, updated_at = ? WHERE id = ?",
                (restored.model_dump_json(), _iso(now), story_id),
            )
            await conn.commit()
        return RestoreStoryVersionResponse(
            restored_story=restored,
            previous_version_id=previous_version_id,
        )

    # --------------------------------------------------------------- sessions

    async def create_session(self, request: CreateSessionRequest) -> StorySession:
        async with self._connect() as conn:
            if request.story_id is not None:
                exists = await _fetchone(
                    conn, "SELECT 1 FROM stories WHERE id = ?", (request.story_id,)
                )
                if exists is None:
                    raise KeyError(request.story_id)
            session = StorySession(
                child_id=request.child_id,
                story_id=request.story_id,
                device_id=request.device_id,
            )
            await conn.execute(
                "INSERT INTO story_sessions (id, child_id, story_id, document, started_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.child_id,
                    session.story_id,
                    session.model_dump_json(),
                    _iso(session.started_at),
                ),
            )
            await conn.commit()
        return session

    async def attach_session_story(self, session_id: str, story_id: str) -> StorySession:
        async with self._connect() as conn:
            session = await _session(conn, session_id)
            updated = session.model_copy(update={"story_id": story_id})
            await conn.execute(
                "UPDATE story_sessions SET story_id = ?, document = ? WHERE id = ?",
                (story_id, updated.model_dump_json(), session_id),
            )
            await conn.commit()
        return updated

    async def get_session(self, session_id: str) -> StorySession | None:
        async with self._connect() as conn:
            row = await _fetchone(
                conn, "SELECT document FROM story_sessions WHERE id = ?", (session_id,)
            )
        return StorySession.model_validate_json(row[0]) if row else None

    async def latest_session_for_child(self, child_id: str) -> StorySession | None:
        """The child's newest session, ended or not.

        A turn that arrives without a session id resolves through this, so
        strikes land on the session the child is actually in rather than on a
        fresh one each turn.
        """
        async with self._connect() as conn:
            row = await _fetchone(
                conn,
                "SELECT document FROM story_sessions WHERE child_id = ? "
                "ORDER BY started_at DESC, rowid DESC LIMIT 1",
                (child_id,),
            )
        return StorySession.model_validate_json(row[0]) if row else None

    async def list_sessions(self) -> list[StorySession]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT document FROM story_sessions ORDER BY started_at DESC"
            )
        return [StorySession.model_validate_json(row[0]) for row in rows]

    async def end_session(
        self,
        session_id: str,
        ended_at: datetime | None = None,
    ) -> StorySession:
        async with self._connect() as conn:
            session = await _session(conn, session_id)
            updated = session.model_copy(update={"ended_at": ended_at or datetime.now(UTC)})
            await conn.execute(
                "UPDATE story_sessions SET document = ? WHERE id = ?",
                (updated.model_dump_json(), session_id),
            )
            await conn.commit()
        return updated

    # ------------------------------------------------------------------ turns

    async def save_conversation_turn(self, turn: ConversationTurn) -> ConversationTurn:
        async with self._connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO conversation_turns "
                "(id, story_id, session_id, document, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    turn.id,
                    turn.story_id,
                    turn.session_id,
                    turn.model_dump_json(),
                    _iso(turn.created_at),
                ),
            )
            await conn.commit()
        return turn

    async def list_conversation_turns(self, limit: int = 50) -> list[ConversationTurn]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT document FROM conversation_turns "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            )
        return [ConversationTurn.model_validate_json(row[0]) for row in rows]

    # ----------------------------------------------------------------- memory

    async def save_memory_chunks(self, chunks: list[MemoryChunk]) -> list[MemoryChunk]:
        if not chunks:
            return []
        async with self._connect() as conn:
            for chunk in chunks:
                await conn.execute(
                    "INSERT OR REPLACE INTO memory_chunks "
                    "(id, story_id, document, created_at) VALUES (?, ?, ?, ?)",
                    (
                        chunk.id,
                        chunk.story_id,
                        chunk.model_dump_json(),
                        _iso(chunk.created_at),
                    ),
                )
                await conn.execute(
                    "DELETE FROM memory_vectors WHERE chunk_id = ?", (chunk.id,)
                )
                if chunk.embedding:
                    await conn.execute(
                        "INSERT INTO memory_vectors (chunk_id, embedding) VALUES (?, ?)",
                        (chunk.id, sqlite_vec.serialize_float32(chunk.embedding)),
                    )
            await conn.commit()
        return chunks

    async def search_memory(
        self,
        query_embedding: list[float],
        story_id: str | None = None,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        if not query_embedding:
            return []
        async with self._connect() as conn:
            # vec0 needs its own k; filter by story afterwards rather than
            # joining inside the match, which vec0 does not support.
            over_fetch = limit if story_id is None else max(limit * 8, 40)
            rows = await conn.execute_fetchall(
                """
                SELECT m.document, v.distance
                FROM (
                    SELECT chunk_id, distance FROM memory_vectors
                    WHERE embedding MATCH ? AND k = ?
                ) AS v
                JOIN memory_chunks AS m ON m.id = v.chunk_id
                ORDER BY v.distance
                """,
                (sqlite_vec.serialize_float32(query_embedding), over_fetch),
            )

        results = []
        for document, distance in rows:
            chunk = MemoryChunk.model_validate_json(document)
            if story_id is not None and chunk.story_id != story_id:
                continue
            results.append(
                MemorySearchResult(
                    id=chunk.id,
                    story_id=chunk.story_id,
                    conversation_turn_id=chunk.conversation_turn_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    # vec0 returns L2 distance. Inverted here so the score is
                    # "higher is nearer", which is what callers order by.
                    score=1.0 / (1.0 + float(distance)),
                    created_at=chunk.created_at,
                )
            )
            if len(results) >= limit:
                break
        return results

    async def delete_memory_chunk(self, memory_id: str) -> bool:
        async with self._connect() as conn:
            cursor = await conn.execute("DELETE FROM memory_chunks WHERE id = ?", (memory_id,))
            await conn.execute("DELETE FROM memory_vectors WHERE chunk_id = ?", (memory_id,))
            await conn.commit()
            return cursor.rowcount > 0

    # ----------------------------------------------------------------- safety

    async def save_safety_event(self, event: SafetyEvent) -> SafetyEvent:
        async with self._connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO safety_events (id, session_id, document, created_at) "
                "VALUES (?, ?, ?, ?)",
                (event.id, event.session_id, event.model_dump_json(), _iso(event.created_at)),
            )
            if event.session_id is not None:
                await _refresh_session_safety_summary(conn, event.session_id)
            await conn.commit()
        return event

    async def list_safety_events(self, limit: int = 50) -> list[SafetyEvent]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT document FROM safety_events ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            )
        return [SafetyEvent.model_validate_json(row[0]) for row in rows]

    async def status_summary(self, child_id: str) -> RepositorySummary:
        """Counts for a polling client, computed in SQL rather than by listing.

        Documents are JSON text, so the per-child filters use json_extract.
        """
        async with self._connect() as conn:
            stories = await _scalar(conn, "SELECT COUNT(*) FROM stories")
            open_sessions = await _scalar(
                conn,
                "SELECT COUNT(*) FROM story_sessions WHERE child_id = ? "
                "AND json_extract(document, '$.ended_at') IS NULL",
                (child_id,),
            )
            unreviewed = await _scalar(
                conn,
                "SELECT COUNT(*) FROM safety_events "
                "WHERE json_extract(document, '$.child_id') = ? "
                "AND json_extract(document, '$.reviewed_at') IS NULL",
                (child_id,),
            )
            pending_media = await _scalar(
                conn,
                "SELECT COUNT(*) FROM media_assets "
                "WHERE json_extract(document, '$.child_id') = ? "
                "AND json_extract(document, '$.status') = ?",
                (child_id, MediaAssetStatus.PENDING_PARENT_REVIEW.value),
            )
        latest = await self.latest_session_for_child(child_id)
        return RepositorySummary(
            stories=stories,
            open_sessions=open_sessions,
            unreviewed_safety_events=unreviewed,
            pending_media=pending_media,
            latest_session_id=latest.id if latest else None,
            latest_session_ended=bool(latest and latest.ended_at is not None),
            latest_session_strikes=latest.safety_summary.get(STRIKE_KEY, 0) if latest else 0,
        )

    async def get_safety_event(self, event_id: str) -> SafetyEvent | None:
        async with self._connect() as conn:
            row = await _fetchone(
                conn, "SELECT document FROM safety_events WHERE id = ?", (event_id,)
            )
        return SafetyEvent.model_validate_json(row[0]) if row else None

    async def release_safety_event(
        self,
        event_id: str,
        request: ReleaseSafetyEventRequest,
    ) -> SafetyEvent:
        """Record a parent overriding a block. Re-running the turn is the caller's job."""
        return await self._update_safety_event(
            event_id,
            {
                "released_at": datetime.now(UTC),
                "released_by": request.released_by,
                "reviewed_at": datetime.now(UTC),
                "reviewed_by": request.released_by,
                "review_note": request.review_note,
            },
        )

    async def mark_safety_event_recapped(self, event_id: str) -> SafetyEvent:
        return await self._update_safety_event(
            event_id, {"recap_delivered_at": datetime.now(UTC)}
        )

    async def _update_safety_event(
        self,
        event_id: str,
        update: dict[str, Any],
    ) -> SafetyEvent:
        async with self._connect() as conn:
            row = await _fetchone(
                conn, "SELECT document FROM safety_events WHERE id = ?", (event_id,)
            )
            if row is None:
                raise KeyError(event_id)
            updated = SafetyEvent.model_validate_json(row[0]).model_copy(update=update)
            await conn.execute(
                "UPDATE safety_events SET document = ? WHERE id = ?",
                (updated.model_dump_json(), event_id),
            )
            await conn.commit()
        return updated

    async def review_safety_event(
        self,
        event_id: str,
        request: ReviewSafetyEventRequest,
    ) -> SafetyEvent:
        async with self._connect() as conn:
            row = await _fetchone(
                conn, "SELECT document FROM safety_events WHERE id = ?", (event_id,)
            )
            if row is None:
                raise KeyError(event_id)
            reviewed = SafetyEvent.model_validate_json(row[0]).model_copy(
                update={
                    "reviewed_at": datetime.now(UTC),
                    "reviewed_by": request.reviewed_by,
                    "review_note": request.review_note,
                }
            )
            await conn.execute(
                "UPDATE safety_events SET document = ? WHERE id = ?",
                (reviewed.model_dump_json(), event_id),
            )
            await conn.commit()
        return reviewed

    # ------------------------------------------------------------------ media

    async def create_media_asset(
        self,
        request: CreateMediaAssetRequest,
        safety: SafetyDecision,
        status: MediaAssetStatus = MediaAssetStatus.PENDING_PARENT_REVIEW,
    ) -> MediaAsset:
        asset = MediaAsset(
            child_id=request.child_id,
            story_id=request.story_id,
            session_id=request.session_id,
            conversation_turn_id=request.conversation_turn_id,
            prompt=request.prompt,
            status=status,
            safety=safety,
        )
        return await self._save_media(asset)

    async def list_media_assets(self, limit: int = 50) -> list[MediaAsset]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT document FROM media_assets ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            )
        return [MediaAsset.model_validate_json(row[0]) for row in rows]

    async def get_media_asset(self, media_id: str) -> MediaAsset | None:
        async with self._connect() as conn:
            row = await _fetchone(
                conn, "SELECT document FROM media_assets WHERE id = ?", (media_id,)
            )
        return MediaAsset.model_validate_json(row[0]) if row else None

    async def approve_media_asset(
        self,
        media_id: str,
        request: ReviewMediaAssetRequest,
    ) -> MediaAsset:
        return await self._update_media(
            media_id,
            {
                "status": MediaAssetStatus.APPROVED,
                "parent_approved_at": datetime.now(UTC),
                "parent_approved_by": request.reviewed_by,
                "parent_note": request.parent_note,
            },
        )

    async def reject_media_asset(
        self,
        media_id: str,
        request: ReviewMediaAssetRequest,
    ) -> MediaAsset:
        return await self._update_media(
            media_id,
            {
                "status": MediaAssetStatus.REJECTED,
                "parent_approved_by": request.reviewed_by,
                "parent_note": request.parent_note,
            },
        )

    async def mark_media_asset_generated(
        self,
        media_id: str,
        *,
        asset_path: str,
        asset_url: str,
        mime_type: str,
        generator: str,
        revised_prompt: str | None = None,
    ) -> MediaAsset:
        update: dict[str, Any] = {
            "status": MediaAssetStatus.GENERATED,
            "asset_path": asset_path,
            "asset_url": asset_url,
            "mime_type": mime_type,
            "generator": generator,
        }
        if revised_prompt is not None:
            update["revised_prompt"] = revised_prompt
        return await self._update_media(media_id, update)

    async def mark_media_asset_failed(self, media_id: str, failure_reason: str) -> MediaAsset:
        return await self._update_media(
            media_id,
            {"status": MediaAssetStatus.FAILED, "failure_reason": failure_reason},
        )

    async def _save_media(self, asset: MediaAsset) -> MediaAsset:
        async with self._connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO media_assets (id, document, created_at) VALUES (?, ?, ?)",
                (asset.id, asset.model_dump_json(), _iso(asset.created_at)),
            )
            await conn.commit()
        return asset

    async def _update_media(self, media_id: str, update: dict[str, Any]) -> MediaAsset:
        current = await self.get_media_asset(media_id)
        if current is None:
            raise KeyError(media_id)
        return await self._save_media(
            current.model_copy(update={**update, "updated_at": datetime.now(UTC)})
        )

    async def save_child_interest(self, interest: ChildInterest) -> ChildInterest:
        """Insert, or update the existing interest with the same name.

        Keeps the original id and created_at on update so the row a summary was
        written against does not move underneath the background lookup.
        """
        existing = await self.get_child_interest(interest.child_id, interest.name)
        if existing is not None:
            interest = interest.model_copy(
                update={
                    "id": existing.id,
                    "created_at": existing.created_at,
                    "updated_at": datetime.now(UTC),
                }
            )
        async with self._connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO child_interests "
                "(id, child_id, name_key, document, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    interest.id,
                    interest.child_id,
                    interest.key,
                    interest.model_dump_json(),
                    _iso(interest.created_at),
                    _iso(interest.updated_at),
                ),
            )
            await conn.commit()
        return interest

    async def get_child_interest(self, child_id: str, name: str) -> ChildInterest | None:
        async with self._connect() as conn:
            row = await _fetchone(
                conn,
                "SELECT document FROM child_interests WHERE child_id = ? AND name_key = ?",
                (child_id, interest_key(name)),
            )
        return ChildInterest.model_validate_json(row[0]) if row else None

    async def list_child_interests(
        self,
        child_id: str,
        limit: int = 20,
    ) -> list[ChildInterest]:
        async with self._connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT document FROM child_interests WHERE child_id = ? "
                "ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (child_id, limit),
            )
        return [ChildInterest.model_validate_json(row[0]) for row in rows]


class _VecConnection:
    """Opens a connection with the sqlite-vec extension loaded."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> aiosqlite.Connection:
        conn = await self._connection
        await conn.enable_load_extension(True)
        await conn.load_extension(sqlite_vec.loadable_path())
        await conn.enable_load_extension(False)
        await conn.execute("PRAGMA foreign_keys = ON")
        return conn

    async def __aexit__(self, *exc: object) -> None:
        await self._connection.__aexit__(*exc)


async def _fetchone(conn: aiosqlite.Connection, sql: str, params: tuple) -> tuple | None:
    cursor = await conn.execute(sql, params)
    return await cursor.fetchone()


async def _scalar(conn: aiosqlite.Connection, sql: str, params: tuple = ()) -> int:
    row = await _fetchone(conn, sql, params)
    return int(row[0]) if row else 0


async def _session(conn: aiosqlite.Connection, session_id: str) -> StorySession:
    row = await _fetchone(conn, "SELECT document FROM story_sessions WHERE id = ?", (session_id,))
    if row is None:
        raise KeyError(session_id)
    return StorySession.model_validate_json(row[0])


async def _refresh_session_safety_summary(conn: aiosqlite.Connection, session_id: str) -> None:
    """Recount the session's safety actions after an event is saved."""
    rows = await conn.execute_fetchall(
        "SELECT document FROM safety_events WHERE session_id = ?", (session_id,)
    )
    summary: dict[str, int] = {}
    for (document,) in rows:
        event = SafetyEvent.model_validate_json(document)
        action = event.safety.action.value
        summary[action] = summary.get(action, 0) + 1
        if event.counted_as_strike:
            summary[STRIKE_KEY] = summary.get(STRIKE_KEY, 0) + 1

    row = await _fetchone(
        conn, "SELECT document FROM story_sessions WHERE id = ?", (session_id,)
    )
    if row is None:
        return
    session = StorySession.model_validate_json(row[0]).model_copy(
        update={"safety_summary": summary}
    )
    await conn.execute(
        "UPDATE story_sessions SET document = ? WHERE id = ?",
        (session.model_dump_json(), session_id),
    )


async def _next_story_id(conn: aiosqlite.Connection, title: str) -> str:
    base = story_id_from_title(title)
    candidate = base
    suffix = 2
    while await _fetchone(conn, "SELECT 1 FROM stories WHERE id = ?", (candidate,)) is not None:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

