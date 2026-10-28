# Yarnotron

Local-first backend for a child-safe AI storytelling companion.

## Home Assistant

This fork (Yarnotron-HA) adds a Home Assistant integration: Assist's wake word,
speech-to-text and text-to-speech are the ears and voice, and Yarnotron stays
the story brain with its safety review, strike lock and parent controls.

- `custom_components/yarnotron/` - the integration (copy it, or add the repo to HACS)
- `blueprints/automation/yarnotron/` - push a notification on a safety alert
- `Dockerfile`, `docker-compose.yml` - run the backend headless on the AI machine
- `docs/yarnotron-ha-guide.html` - architecture, code changes, setup and user guide
- `src/yarnotron/supervisor/`, `supervisor.example.toml` - Mac-side service so Home Assistant can launch, shut down and check the Mac's processes (`docs/mac-power-control.md`)
- `integrations/hermes/` - Hermes parent/operator profile and stdio MCP server for Yarnotron (`docs/yarnotron-for-jeevus.md` describes the whole system for the Jeevus project)
- `ha_tests/` - integration tests (`pip install -r requirements_test_ha.txt`, then `pytest ha_tests`)

Set `YARNOTRON_API_TOKEN` on the backend whenever it is reachable from the
network. The Home Assistant side is tested against Home Assistant 2026.2 only,
not against a physical voice satellite.

## Quick Start

```sh
uv sync --extra dev --extra voice
python3 scripts/setup_yarnotron.py --install   # checks Ollama, fetches voice models
python3 scripts/start_yarnotron.py --voice
```

`setup_yarnotron.py` reports every prerequisite as ok or missing with the
command that fixes it, and downloads the Kokoro voice files. Run it any time
something stops working and you want to know which part is missing.

The API will be available at:

```text
http://127.0.0.1:8989
```

The parent dashboard will be available at:

```text
http://127.0.0.1:8989/dashboard
```

Stop local services:

```sh
python3 scripts/stop_yarnotron.py
```

The startup script runs dependency sync when `--sync` is provided, starts the
FastAPI backend in the background, waits for `/health`, and speaks:

```text
Welcome to Yarnotron, what's your story?
```

Use `--no-welcome` to skip the spoken startup greeting, `--welcome-message` to
change it, or `--welcome-voice` to choose a macOS `say` voice.

To start the microphone listener too:

```sh
python3 scripts/start_yarnotron.py --voice --story-id story_clock_tower_mystery
```

You can also omit `--story-id` and let the first story turn create or resume a
story. If the child clearly starts a new story, Yarnotron creates a readable
id from the title, such as `story_clock_tower_mystery`, and reminds them to
remember the title. If the first turn sounds like it might be a previous story,
Yarnotron briefly asks whether this is a new story or a continuation. You can
restart voice sessions with either the readable id or the exact title, such as
`--story-id "Clock Tower Mystery"`.

When `--voice` is running, the local voice loop waits for the wake word
`interpreter`. After it hears the wake word, it says "I'm listening. What's your
story?", records story turns, sends each turn to the backend, speaks the
response, then listens again. If it hears no speech for 10 seconds, it returns to
wake-word mode.

The listener also supports a small set of local voice controls:

- `help` or `what can I say?` gives examples.
- `repeat that` replays the last assistant response.
- `stop listening` returns to wake-word mode.
- `pause` waits until you say `interpreter resume`.
- `end session` cleanly closes the voice session.

The shutdown script stops the background backend.

Useful endpoints:

- `GET /health`
- `GET /v1/models/ollama`
- `POST /v1/stories`
- `GET /v1/stories`
- `POST /v1/agent/turn`
- `POST /v1/memory/search`

## Ollama

The backend expects Ollama at `http://127.0.0.1:11434` by default.

Override with:

```sh
export YARNOTRON_OLLAMA_BASE_URL=http://127.0.0.1:11434
export YARNOTRON_DEFAULT_MODEL=gemma4:12b-mlx
export YARNOTRON_SAFETY_MODEL=gemma4:e2b-mlx
export YARNOTRON_EMBEDDING_MODEL=nomic-embed-text:v1.5
export YARNOTRON_EMBEDDING_DIMENSIONS=768
# Stories live in a SQLite file under ~/Library/Application Support/Yarnotron/
# by default. Override the whole data directory with YARNOTRON_DATA_DIR, or
# point YARNOTRON_DATABASE_URL at a specific file.
```

Yarnotron defaults to `nomic-embed-text:v1.5` with 768-dimensional vectors.
When no embedding model is configured, Yarnotron uses a deterministic local
fallback embedding so memory search works in development.

Safety review is layered:

- a fast deterministic keyword guard,
- a local Ollama JSON safety reviewer for child transcripts,
- a second local Ollama JSON safety reviewer for assistant responses and proposed story patches.

If `YARNOTRON_SAFETY_MODEL` is empty, the default chat model is reused for
safety review, which works but costs about 1 s per review. `gemma4:e2b-mlx` is
roughly 2.8x faster with near-identical judgement.

Do not pick a safety model without measuring it. `scripts/bench_safety_models.py`
scores candidates on JSON validity, latency, and whether they actually catch
unsafe content — `qwen3.5:0.8b-mlx` looks ideal on the first two and allows
essentially everything. See `docs/phase-0-baseline.md`.

Story editing is policy-checked after the model drafts structured patches:

- destructive edits are proposed but not applied until the child confirms clearly,
- duplicate characters and scenes are merged into existing story items,
- the child's spoken wording is preserved in scene draft text when it belongs in the story,
- explicit requests like "undo last change" restore the most recent saved story version.

Search story memory:

```sh
curl -s -X POST http://127.0.0.1:8989/v1/memory/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"moonlight music box midnight","limit":3}'
```

Open the local parent dashboard:

```text
http://127.0.0.1:8989/dashboard

To sign in with a username and password instead of the API token, set
`YARNOTRON_DASHBOARD_PASSWORD` (and optionally `YARNOTRON_DASHBOARD_USER`, default `parent`).
The browser then shows its own sign-in box for `/dashboard`, and the same login opens `/v1`.
The bearer token keeps working for Home Assistant. Both travel unencrypted on a plain-HTTP LAN.
```

The dashboard includes parent controls for exporting or deleting stories,
reviewing and restoring story versions, ending active sessions, marking safety
events reviewed, approving generated image requests, generating approved local
image placeholders, selecting a preferred voice, playing a quick voice sample,
and removing individual memory chunks from search results.

Generated images are safety gated:

- a child image request creates a pending media item instead of generating immediately,
- the prompt must pass safety review before it enters the parent review queue,
- a parent must approve the media item from the dashboard or API,
- generation performs a final prompt safety review before writing a local image file.

The default generator writes an SVG placeholder to `generated_media/` and serves
it from `/media/...`. This keeps the workflow local and testable until a heavier
local image backend is plugged in.

## Local Voice Prototype

The first real voice path is a local CLI loop:

```text
microphone -> RMS voice activity detection -> faster-whisper -> FastAPI backend -> pluggable TTS
```

It keeps the backend contract unchanged. The loop still sends the transcript to
`POST /v1/agent/turn` and speaks the returned `response_text`.

Install the voice dependencies:

```sh
python3 scripts/start_yarnotron.py --sync
```

Start the backend services and microphone listener:

```sh
python3 scripts/start_yarnotron.py --voice --story-id story_clock_tower_mystery
```

Omit `--story-id` to have Yarnotron create or resolve the story from the
first spoken story turn.

Or start only the backend services and run the local microphone loop in another terminal:

```sh
python3 scripts/start_yarnotron.py
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --story-id story_clock_tower_mystery
```

The loop records from the microphone, uses RMS-based voice activity detection,
transcribes with `faster-whisper`, sends turns to the backend, and speaks replies
through a selectable TTS provider. On this Mac, the default `auto` TTS provider
uses the local macOS `say` binary as the simplest working speaker path. The first
run downloads the configured Whisper model unless it is already cached locally.

Say `interpreter` to wake the listener. It returns to wake-word mode after 10
seconds of silence. Tune this with `--wake-word`, `--wake-response`, and
`--idle-timeout-seconds`.

Simple controls are handled locally for responsiveness: say `help`, `repeat
that`, `stop listening`, `pause`, `interpreter resume`, or `end session`.

Turn timing is written to the voice and backend logs as JSON-like lines prefixed
with `timing`. Check `.yarnotron/logs/voice.log` for microphone, transcription,
backend round trip, sound-effect, and TTS timing. Check
`.yarnotron/logs/backend.log` for safety review, memory search, model
generation, patching, persistence, and memory embedding timing.

Use Ctrl-C to end the session cleanly.

List audio devices:

```sh
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --list-audio-devices
```

Choose a microphone:

```sh
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py \
  --story-id story_clock_tower_mystery \
  --input-device 0
```

TTS providers:

- `auto`: uses macOS `say` when available, otherwise no speech.
- `none`: prints responses only.
- `macos-say`: uses a selected macOS voice.
- `piper`: uses a local Piper voice model.

The parent dashboard can save a preferred voice under Voice Preferences. The
local voice loop uses that saved voice unless `--tts-voice` is provided.

macOS voice selection:

```sh
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --list-macos-voices
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py \
  --story-id story_clock_tower_mystery \
  --tts-provider macos-say \
  --tts-voice Samantha
```

Piper voice selection:

```sh
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py \
  --story-id story_clock_tower_mystery \
  --tts-provider piper \
  --piper-model models/piper/en_US-lessac-medium.onnx
```

Platform notes:

- macOS: `brew install portaudio ffmpeg`
- Linux: install PortAudio and FFmpeg, for example `sudo apt install portaudio19-dev ffmpeg`
- Windows: install FFmpeg and use the `sounddevice` wheel; Piper can be used as an external binary.

`faster-whisper` is speech-to-text only. Voice selection belongs to the TTS
provider, such as macOS `say` voices or Piper `.onnx` voice models.

## Interrupting

Yarnotron listens while it speaks, so a child can talk over a reply. On speakers
the microphone also hears Yarnotron, so the detector calibrates against its own
echo and looks for something clearly louder and sustained. How much louder
depends on the room, and it is tunable:

```sh
python3 scripts/start_yarnotron.py --voice --no-barge-in          # never interrupt
python3 scripts/start_yarnotron.py --voice --barge-in-margin 1.8  # noisy room
python3 scripts/start_yarnotron.py --voice --barge-in-frames 10   # must be sustained
```

If replies are being cut off mid-sentence, raise the margin or turn it off.
Headphones remove the problem entirely.

## Safety tiers

The reviewer sees the story's title and who is in it, not just the sentence.
A line is often innocent only in context — "she deflates the toys sometimes when
she bites them" reads very differently once you know "she" is a dog. Reviewed
bare, `gemma4:12b` blocked exactly that as sexual content. Only names and roles
are sent, never scene text or the child's earlier lines: context is there to
explain, and must not become something a story can be written to exploit.

Two tiers, and they behave differently on purpose.

**Serious content** — sexual material, self-harm, anything the reviewer returns
as `block` or `alert_parent` — stops the turn on the first mention, saves a note
for the parent, and writes nothing to the story.

**Milder content** — the `redirect` tier, things like `steal`, `gore` or `bully`
— no longer ends the turn. The child hears a short caution, the parent still
gets the event, and the story carries on. Being stopped mid-sentence teaches a
child not to talk rather than what is safe.

Each cautioned turn spends a strike. After
`YARNOTRON_SAFETY_STRIKE_LIMIT` of them (default 5) the session ends with a calm
message and further turns are refused without even calling the safety model.
Strikes are counted per session on `StorySession.safety_summary`, so starting a
new session begins again at zero — it is a cooling-off, not a lockout. A
redirect caused by the *assistant's* own output is recorded but never counted as
the child's strike.

A block stops the turn and writes nothing, so a wrong one costs the child their
idea. The parent dashboard has a **Release block** button on those events: it
re-runs the turn so the idea lands in the story, and the next time the child
opens that story Yarnotron reads the line back and says that is where they left
off — they were not listening when it ran. The released line stays allowed for
the rest of that session, so the child can revisit it, and is blocked again in
any later session. Only a block can be released; a caution already let the story
continue, and a session-ended refusal decided nothing about what was said.

A turn does not have to name a session. One that omits `session_id` attaches to
the child's most recent session: a new one is created if there is none, an open
one keeps accumulating, and an **ended** one still refuses the turn. Leaving the
field out is therefore not a way to start over — that takes
`POST /v1/sessions`, the deliberate act the cut-off message asks for. Every
response carries the `session_id` it landed in, so a caller can keep the next
turn in the same one.

## Development

```sh
python3 scripts/start_yarnotron.py --sync
uv run pytest
uv run ruff check .
python3 scripts/stop_yarnotron.py
```

Stories live in a SQLite file, so the whole suite runs with no server:
`tests/test_sqlite_story_repository.py` covers sessions, safety events, version
restore, memory search, interests and patch persistence.

If you still have stories in the old Postgres database, bring them across once
with the migration script. It is the only thing left that needs `psycopg`:

```bash
uv sync --extra postgres
uv run python scripts/migrate_postgres_to_sqlite.py \
  --from postgresql://yarnotron:yarnotron@127.0.0.1:5433/yarnotron
```

If your shell already has the project installed in editable mode, `PYTHONPATH=src` is not required. Keeping it in the command makes the local setup robust on machines with spaces in the project path.
