# Yarnotron One-Page User Manual

Yarnotron is a local-first storytelling backend for child-safe AI writing sessions. It stores editable story state, reviews turns for safety, supports a parent dashboard, and can run an optional local microphone/speaker loop.

## 1. Start Yarnotron

Prerequisites: `uv`, Ollama, and any configured local Ollama models. For voice mode, also install the platform audio tools noted in `README.md`.

```sh
uv sync --extra dev
uv sync --extra dev --extra voice
PYTHONPATH=src uv run uvicorn yarnotron.main:app --reload
```

Open the backend at `http://127.0.0.1:8989`. Check that it is alive with `GET /health`, then open the parent dashboard at `http://127.0.0.1:8989/dashboard`.

Optional environment overrides can go in `.env`:

```sh
YARNOTRON_OLLAMA_BASE_URL=http://127.0.0.1:11434
YARNOTRON_DEFAULT_MODEL=gemma4:12b-mlx
YARNOTRON_SAFETY_MODEL=gemma4:e2b-mlx
YARNOTRON_EMBEDDING_MODEL=nomic-embed-text:v1.5
YARNOTRON_DATABASE_URL=~/Library/Application Support/Yarnotron/yarnotron.db
YARNOTRON_MEDIA_OUTPUT_DIR=generated_media
```

## 2. Create and Continue a Story

Create a story:

```sh
curl -s -X POST http://127.0.0.1:8989/v1/stories \
  -H 'Content-Type: application/json' \
  -d '{"title":"The Moon Clock","synopsis":"A child inventor finds a clock that only ticks under moonlight."}'
```

Copy the returned `id`; for this title it will be `story_the_moon_clock`. Keep
the title in mind so you can resume the same story in a future voice session.
You can also skip this manual story creation for voice sessions: if the first
spoken story turn clearly starts a new story, Yarnotron creates the title and
id automatically. Then send a storytelling turn:

```sh
curl -s -X POST http://127.0.0.1:8989/v1/agent/turn \
  -H 'Content-Type: application/json' \
  -d '{"child_id":"local_child","story_id":"story_the_moon_clock","transcript":"Add a friendly robot who guards the moon clock."}'
```

The response includes the assistant reply, safety decision, proposed patches, applied patches, updated story, memory IDs, and media request IDs when relevant.

## 3. Use the Parent Dashboard

Use `http://127.0.0.1:8989/dashboard` to:

- refresh and inspect stories, sessions, media, safety events, and searchable memories,
- export a story as Markdown,
- review and restore saved story versions,
- end active sessions,
- mark safety events reviewed,
- approve, reject, and generate pending image requests,
- remove individual memory chunks from search.

Generated image requests are parent-gated. A child request creates a pending media item, safety review runs first, a parent approves it, and generation writes a local placeholder image under `generated_media/` that is served from `/media/...`.

## 4. Run the Local Voice Loop

Start the backend first, then run:

```sh
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --story-id story_the_moon_clock
```

Omit `--story-id` to let the first spoken story turn create or resume a story.
If Yarnotron cannot tell whether it is new or previous, it asks one brief
clarifying question. The loop records microphone input, uses local
speech-to-text, sends the transcript to `POST /v1/agent/turn`, and speaks the
returned `response_text` with the selected TTS provider. Press `Ctrl-C` to end
cleanly.

Helpful voice commands:

```sh
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --list-audio-devices
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --list-macos-voices
PYTHONPATH=src uv run --extra voice scripts/local_voice_loop.py --story-id story_the_moon_clock --tts-provider none
```

## Troubleshooting

- Backend unavailable: confirm `uvicorn` is still active.
- Model errors: confirm Ollama is running and the configured chat, safety, and embedding models are available.
- Voice errors: install PortAudio and FFmpeg, then rerun `uv sync --extra dev --extra voice`.
- Empty memory search: Yarnotron uses the configured embedding model when available and a deterministic local fallback during development.
