"""Open the story repository.

The datastore is a SQLite file. This stays a separate module from
`sqlite_story_repository` so the app has one place to ask for a ready
repository without caring how it is built.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from yarnotron.core.config import Settings
from yarnotron.services.sqlite_story_repository import SqliteStoryRepository
from yarnotron.services.story_store import StoryRepository


@asynccontextmanager
async def open_story_repository(settings: Settings) -> AsyncIterator[StoryRepository]:
    """Open the repository, initialised and ready, and close it after."""
    if settings.database_url.startswith("postgres"):
        # Anyone upgrading may still have this in a .env. Say so, rather than
        # treating the URL as a filename and quietly creating a file called
        # "postgresql:".
        raise ValueError(
            "YARNOTRON_DATABASE_URL points at Postgres, which Yarnotron no longer "
            "supports. Point it at a SQLite file, or bring an existing database "
            "across with scripts/migrate_postgres_to_sqlite.py first."
        )

    repository = SqliteStoryRepository(settings)
    await repository.initialize()
    yield repository
