# Phase 3: delete Docker

Make the datastore a file, so the app has nothing to install alongside it.

## What shipped

| item | status |
|---|---|
| `SqliteStoryRepository` against the same protocol, sqlite-vec for memory | done |
| Postgres implementation and its suite kept, both asserted against | done, since removed |
| one-shot migration for existing stories | done, and run |
| data under an application-support directory | done |
| `docker-compose.yml` and the Docker branches deleted | done |

**Exit criteria met.** Full suite green on SQLite, `docker` appears nowhere
outside historical docs and the migration tool, and the app runs on a machine
with no Docker, no Postgres driver and no audio libraries.

## Storage was never the bottleneck, and still isn't

Full 56-turn corpus, streamed, seed 42, `gemma4:e2b-mlx` safety, same session:

| stage | Postgres | SQLite |
|---|---:|---:|
| **total turn p50** | **6473 ms** | **6456 ms** |
| patch_apply | 5.1 | 3.9 |
| save_turn | 2.1 | 3.0 |
| save_memory_chunks | 3.1 | 3.8 |
| memory_search | 2.9 | 3.6 |
| story_lookup | 1.9 | 2.2 |
| save_safety_event | 2.0 | 3.6 |
| **all storage per turn** | **23.6 ms** | **25.0 ms** |

Both 56/56 with no failures and identical intent routing. Storage is 0.4% of a
turn either way, which is the same thing phase 0 found: three model calls are
99.6% of the work. Swapping the database changes nothing a child can perceive —
it changes what a parent has to install.

SQLite opens a connection per operation rather than pooling. At 25 ms of
storage in a 6.5 s turn that is not worth fixing, and a pool would need care
around SQLite's writer lock. Noted rather than optimised.

## The exit criterion earned its keep

"The app runs on a machine with Docker uninstalled" sounds like a formality.
Tried properly — `uv venv` and `pip install -e .` into an empty environment —
it failed twice, on things the test suite could not see because the dev
environment has everything installed.

**psycopg was mandatory to import the app.** `story_repository` imported it at
module scope, and every route and the turn processor imported the
`StoryRepository` protocol from that module. Knowing what a repository *is*
required the Postgres driver. The protocol, `StoryPatchApplier` and
`story_id_from_title` now live in `story_store`, which imports nothing heavy;
`story_repository` keeps the Postgres implementation and re-exports them.

**The backend depended on the voice extra.** Wiring Kokoro into the preferences
service in phase 2 meant importing the app pulled in numpy and sounddevice — an
API server needing audio libraries to start. Those imports are now inside the
two functions that use them. This was a phase 2 regression that phase 3's exit
criterion caught.

Both are the same shape of bug: an import at module scope that quietly widens
what has to be installed. Worth testing for directly before phase 6 tries to
package this.

## Migration

`scripts/migrate_postgres_to_sqlite.py`, run against the live database rather
than only written: 77 rows across 8 tables. Verified afterwards rather than
trusting the row counts —

- story content intact, characters and facts present
- version chronology correct, oldest snapshot holds the pre-patch state
- vector search working on migrated embeddings, exact match scoring 1.000
- the file still takes new writes, adding a fact and a new version

It reads rows directly rather than going through the repository, so ids,
timestamps and version history survive, and it refuses to write into a file
that already holds stories unless `--force` is given.

## Where the data lives

`~/Library/Application Support/Yarnotron/` by default — database, preferences
and generated media — rather than `.yarnotron/` inside the checkout. An
installed app cannot write into the source tree it was built from, and a
child's stories should survive deleting the repository. `YARNOTRON_DATA_DIR`
overrides the directory; `YARNOTRON_DATABASE_URL` still takes a specific file
or a specific file.

## Keeping Postgres — since undone

Phase 3 kept the Postgres implementation and ran both suites against the same
protocol, which is what made this a proved port rather than a rewrite:
`tests/test_sqlite_story_repository.py` is the Postgres suite rewritten against
SQLite, plus coverage the original lacked — media lifecycle, memory chunk
deletion, and that data survives reopening the file.

**Postgres was removed afterwards.** Having served its purpose as the thing the
port was proved against, `PostgresStoryRepository` and its suite were deleted;
`YARNOTRON_DATABASE_URL` now takes a file path only and rejects a
`postgresql://` URL with a message pointing at the migration script. The one
part still standing is `scripts/migrate_postgres_to_sqlite.py`, for anyone who
never made the trip, and the optional `postgres` extra exists only to install
`psycopg` for it.

## A portability risk for phase 6

sqlite-vec is a loadable extension, and macOS system Python is commonly built
with `enable_load_extension` disabled. It works here on the uv-managed Python.
Phase 6 bundles its own runtime, so this needs checking against whatever
interpreter Tauri ships with, not assumed from this machine.
