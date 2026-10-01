"""One-shot migration of an existing Postgres database into the SQLite file.

Phase 3 makes the datastore a file. Anyone who has been using Yarnotron already
has stories in Postgres, and they should not lose them.

    docker compose -p yarnotron up -d          # the old database, still running
    uv run python scripts/migrate_postgres_to_sqlite.py \
        --from postgresql://yarnotron:yarnotron@127.0.0.1:5433/yarnotron

Reads rows directly rather than going through the repository, so ids,
timestamps and version history survive intact. Refuses to write into a file
that already has stories unless --force is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.core.config import Settings, get_settings
from yarnotron.models.media import MediaAsset
from yarnotron.models.memory import ConversationTurn, MemoryChunk
from yarnotron.models.operations import SafetyEvent, StorySession
from yarnotron.models.story import Story
from yarnotron.services.sqlite_story_repository import SqliteStoryRepository, database_path

TABLES = (
    "stories",
    "story_versions",
    "story_patches",
    "story_sessions",
    "conversation_turns",
    "safety_events",
    "memory_chunks",
    "media_assets",
)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat()


async def migrate(source_url: str, target: SqliteStoryRepository, force: bool) -> dict[str, int]:
    counts = dict.fromkeys(TABLES, 0)

    async with target._connect() as sqlite:  # noqa: SLF001 - one-shot tool
        existing = await (await sqlite.execute("SELECT count(*) FROM stories")).fetchone()
        if existing[0] and not force:
            raise SystemExit(
                f"target already holds {existing[0]} stories; pass --force to write anyway"
            )

        async with await psycopg.AsyncConnection.connect(source_url, row_factory=dict_row) as pg:
            for row in await (await pg.execute(
                "SELECT id, document, created_at, updated_at FROM stories"
            )).fetchall():
                story = Story.model_validate(row["document"])
                await sqlite.execute(
                    "INSERT OR REPLACE INTO stories (id, document, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (row["id"], story.model_dump_json(), _iso(row["created_at"]),
                     _iso(row["updated_at"])),
                )
                counts["stories"] += 1

            for row in await (await pg.execute(
                "SELECT id, story_id, document, created_at FROM story_versions"
            )).fetchall():
                await sqlite.execute(
                    "INSERT OR REPLACE INTO story_versions (id, story_id, document, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (row["id"], row["story_id"],
                     Story.model_validate(row["document"]).model_dump_json(),
                     _iso(row["created_at"])),
                )
                counts["story_versions"] += 1

            for row in await (await pg.execute(
                "SELECT id, story_id, operation, data, child_confirmation_required, rationale, "
                "created_at FROM story_patches"
            )).fetchall():
                await sqlite.execute(
                    "INSERT OR REPLACE INTO story_patches (id, story_id, operation, data, "
                    "child_confirmation_required, rationale, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row["id"], row["story_id"], row["operation"], json.dumps(row["data"]),
                     int(bool(row["child_confirmation_required"])), row["rationale"],
                     _iso(row["created_at"])),
                )
                counts["story_patches"] += 1

            for row in await (await pg.execute(
                "SELECT id, child_id, story_id, device_id, started_at, ended_at, safety_summary "
                "FROM story_sessions"
            )).fetchall():
                session = StorySession(
                    id=row["id"], child_id=row["child_id"], story_id=row["story_id"],
                    device_id=row["device_id"], started_at=row["started_at"],
                    ended_at=row["ended_at"], safety_summary=row["safety_summary"] or {},
                )
                await sqlite.execute(
                    "INSERT OR REPLACE INTO story_sessions (id, child_id, story_id, document, "
                    "started_at) VALUES (?, ?, ?, ?, ?)",
                    (session.id, session.child_id, session.story_id,
                     session.model_dump_json(), _iso(session.started_at)),
                )
                counts["story_sessions"] += 1

            for row in await (await pg.execute(
                "SELECT id, story_id, session_id, document FROM conversation_turns"
            )).fetchall():
                turn = ConversationTurn.model_validate(row["document"])
                await sqlite.execute(
                    "INSERT OR REPLACE INTO conversation_turns (id, story_id, session_id, "
                    "document, created_at) VALUES (?, ?, ?, ?, ?)",
                    (turn.id, row["story_id"], row["session_id"], turn.model_dump_json(),
                     _iso(turn.created_at)),
                )
                counts["conversation_turns"] += 1

            for row in await (await pg.execute(
                "SELECT id, child_id, story_id, session_id, conversation_turn_id, safety, "
                "transcript_excerpt, parent_alert_sent, reviewed_at, reviewed_by, review_note, "
                "created_at FROM safety_events"
            )).fetchall():
                event = SafetyEvent(
                    id=row["id"], child_id=row["child_id"], story_id=row["story_id"],
                    session_id=row["session_id"],
                    conversation_turn_id=row["conversation_turn_id"],
                    safety=row["safety"], transcript_excerpt=row["transcript_excerpt"],
                    parent_alert_sent=bool(row["parent_alert_sent"]),
                    reviewed_at=row["reviewed_at"], reviewed_by=row["reviewed_by"],
                    review_note=row["review_note"], created_at=row["created_at"],
                )
                await sqlite.execute(
                    "INSERT OR REPLACE INTO safety_events (id, session_id, document, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (event.id, event.session_id, event.model_dump_json(), _iso(event.created_at)),
                )
                counts["safety_events"] += 1

            for row in await (await pg.execute(
                "SELECT id, story_id, conversation_turn_id, content, embedding, metadata, "
                "created_at FROM memory_chunks"
            )).fetchall():
                embedding = row["embedding"]
                if isinstance(embedding, str):  # pgvector renders as '[1,2,3]'
                    embedding = json.loads(embedding)
                chunk = MemoryChunk(
                    id=row["id"], story_id=row["story_id"],
                    conversation_turn_id=row["conversation_turn_id"],
                    content=row["content"],
                    embedding=list(embedding) if embedding is not None else None,
                    metadata=row["metadata"] or {}, created_at=row["created_at"],
                )
                await sqlite.execute(
                    "INSERT OR REPLACE INTO memory_chunks (id, story_id, document, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk.id, chunk.story_id, chunk.model_dump_json(), _iso(chunk.created_at)),
                )
                if chunk.embedding:
                    import sqlite_vec

                    await sqlite.execute(
                        "DELETE FROM memory_vectors WHERE chunk_id = ?", (chunk.id,)
                    )
                    await sqlite.execute(
                        "INSERT INTO memory_vectors (chunk_id, embedding) VALUES (?, ?)",
                        (chunk.id, sqlite_vec.serialize_float32(chunk.embedding)),
                    )
                counts["memory_chunks"] += 1

            for row in await (await pg.execute("SELECT * FROM media_assets")).fetchall():
                asset = MediaAsset.model_validate(
                    {k: v for k, v in row.items() if k in MediaAsset.model_fields}
                )
                await sqlite.execute(
                    "INSERT OR REPLACE INTO media_assets (id, document, created_at) "
                    "VALUES (?, ?, ?)",
                    (asset.id, asset.model_dump_json(), _iso(asset.created_at)),
                )
                counts["media_assets"] += 1

        await sqlite.commit()
    return counts


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="source", required=True, help="postgresql:// URL to read from"
    )
    parser.add_argument("--to", default=None, help="sqlite file to write (default: configured)")
    parser.add_argument("--force", action="store_true", help="write even if the target has data")
    args = parser.parse_args()

    settings = get_settings() if args.to is None else Settings(database_url=args.to)
    target_path = database_path(settings)
    repository = SqliteStoryRepository(settings, path=target_path)
    await repository.initialize()

    print(f"from {args.source}\n  to {target_path}\n")
    counts = await migrate(args.source, repository, args.force)

    width = max(len(name) for name in counts)
    for name, count in counts.items():
        print(f"  {name.ljust(width)}  {count:>6}")
    print(f"\nmigrated {sum(counts.values())} rows")

    stories = await repository.list()
    print(f"target now holds {len(stories)} stories")


if __name__ == "__main__":
    asyncio.run(main())
