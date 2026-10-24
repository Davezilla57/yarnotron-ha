# Yarnotron: reference for the Jeevus project

Purpose: describe everything about Yarnotron that matters for making it a component of Jeevus: what it is, what must not change, every interface it offers, how it connects to Home Assistant, and where the open decisions are. Facts were read from the code in this repository (branch `feature/hermes-profile`) unless marked **unverified**.

---

## 1. What Yarnotron is

A voice storytelling companion for a child. The child talks; Yarnotron continues, edits and remembers a story, explains words, and offers to draw pictures. It is built around three commitments:

1. **Safety is layered and deterministic.** A keyword guard, a model reviewer, a per-sentence output gate and a strike limit sit between the child and the model. They are code, not prompt text.
2. **The child's words are the child's.** Stories store the child's wording verbatim; edits that remove or overwrite need the child's confirmation; parents see alerts, not transcripts, unless they opt in.
3. **A grown-up is always in control.** Strike-limit sessions stay locked until a parent starts a new one; held-back content is released only by a parent; pictures wait for parent approval.

It started as a local microphone prototype (Mac + faster-whisper + Kokoro). It now has a second front end: **Home Assistant Assist**, which owns wake word, speech-to-text and text-to-speech, and talks to the same backend. Everything below describes that second arrangement; section 8 notes the older one.

## 2. System map

```
                         Home Assistant OS (VM on Proxmox / Dell R620)
 ┌──────────────┐        ┌───────────────────────────────────────────────────────────┐
 │ Voice        │ audio  │ Assist pipeline: wake word → STT (Wyoming Whisper) →       │
 │ satellite    │◀──────▶│   conversation agent "Yarnotron" → TTS (Wyoming Piper)     │
 └──────────────┘        │ custom_components/yarnotron: agent, sensors, buttons,      │
                         │   services, events, session lock, power control            │
                         └──────┬───────────────────────┬────────────────────────────┘
                   text turns   │ HTTP :8989 (bearer)    │ HTTP :8990 (bearer)
                                ▼                        ▼
 MacBook (M5 Max)   ┌────────────────────┐    ┌──────────────────────┐
                    │ Yarnotron backend  │    │ Mac supervisor       │  launchd, stdlib only
                    │ FastAPI + SQLite   │◀───│ check / start / stop │──▶ starts and stops:
                    └─────────┬──────────┘    └──────────────────────┘     Ollama :11434
                              │ HTTP                                       Whisper (Wyoming) :10300
                              ▼                                            backend :8989
                    Ollama: story model, safety model, embeddings

 Piper (TTS) runs as a Home Assistant add-on. Whisper (STT) runs on the Mac.
 Optional: Hermes profile ──MCP──▶ yarnotron_mcp.py ──▶ HA services / backend   (section 10)
```

Hosts and ports (defaults): backend `8989`, supervisor `8990`, Ollama `11434`, Whisper Wyoming `10300`, Home Assistant `8123`.

## 3. The turn, step by step (backend)

`src/yarnotron/agent/turn_processor.py`, `TurnProcessor.process`:

1. **Session.** Use the given `session_id`, else the child's latest, else create one. An ended session refuses the turn with a fixed message and category `session_ended` (a parent release bypasses this).
2. **Story load.** Unknown `story_id` → 404. Loaded before safety review so the reviewer sees title and character names.
3. **Release lookup.** Recent safety events are checked for a parent-released identical transcript in this session.
4. **Intent routing** (`agent/router.py`, keyword/substring, first match wins): unclear filler → `UNKNOWN`; undo; definition; grammar; picture/draw → `CREATE_IMAGE`; change/remove/add/rename → `EDIT_STORY`; parent words → `PARENT_COMMAND`; else `CONTINUE_STORY`.
5. **Input keyword guard** (whole-word). *Alert tier* (self-harm and sexual terms) → `ALERT_PARENT`, turn stopped. *Redirect tier* (gore, torture, drugs, stealing, bullying…) → caution plus a strike, story continues. "hate" is deliberately excluded.
6. **Confirmation resolve.** A yes/no to the previous turn's pending destructive patches or interest question is resolved. Pending state is derived from the stored previous turn, so it survives restarts.
7. **Short-circuits.** Story list, story switch, unknown, undo (restores the latest version snapshot), image request (saves a pending media asset).
8. **First-turn story selection.** Pick an existing story, create one, or ask "new or continue?".
9. **Memory retrieval.** Embed the transcript; top 5 memory chunks for the story; the child's interests.
10. **Model call.** Prompt from a compiled, bounded story view + memories + interests. Buffered: `generate_json`. Streamed: `stream_json`, releasing each sentence through the output gate (one failed sentence vetoes further speech). Draft shape: `response_text`, `story_patches`, `tutor_notes`, `interest_candidate`.
11. **Editing policy** (`agent/editing_policy.py`). Merges duplicate adds, preserves the child's wording; `remove_*`, `update_*` without an explicit edit word, and anything that clears text need child confirmation and are held.
12. **Interests.** Ambiguous → hold patches and ask where it belongs. Unambiguous → record (deduped) and enrich in a background task (≤300 chars, through the output gate).
13. **Output safety.** Full review of text + patches. Block/alert → fixed stop message, nothing applied. Redirect → the reviewer's child-facing message. Otherwise apply patches (each snapshots a version first).
14. **Persist.** Conversation turn, safety event if flagged, one memory chunk (only with a story), embedding.
15. **Deferred input review.** After replying, `ModelSafetyReviewer.review_transcript` runs in the background for allowed turns. It cannot stop the turn; it records a safety event (counting a strike for redirect, alerting a parent for block/alert).

### Streaming

`POST /v1/agent/turn/stream` returns NDJSON: `{"sentence": …}` per gated sentence, then one `{"turn": {…}}`, or `{"error": …, "status": …}`. Paths that never reach the model stream nothing; the client speaks `turn.response_text`.

## 4. Safety model

| Layer | Where | Effect |
|---|---|---|
| Keyword guard | before the model | alert-tier stops; redirect-tier cautions |
| Deferred model review | after the reply | records event; strike or parent alert; fails open to `allow` with category `safety_model_invalid_output` after one retry (Ollama down is not swallowed) |
| Sentence gate | during streaming | vetoes further speech from one failed sentence |
| Output review | before applying patches | blocks/redirects |
| Media review | at request and before generation | non-allow requests are stored as `rejected` |
| Interest summaries | background | through the output gate |

Actions: `allow`, `redirect`, `block`, `alert_parent`. Severities: `none`, `info`, `review`, `alert`, `urgent`. Combined decisions take the highest risk rank.

**Strikes.** A session keeps a strike count. Redirect decisions that are flagged `counted_as_strike` add one. At `safety_strike_limit` (default 5) the session ends; the ending turn is stored with category `strike_limit_reached`. Block and alert never spend a strike (they stop the turn). Only a new session (`POST /v1/sessions`) clears the state.

**Parent release.** `POST /v1/safety-events/{id}/release` re-runs a blocked turn as parent-released; that line stays allowed for the session, and the child later hears a recap. Not available for `session_ended` events or events without a story.

## 5. Data and persistence

- **Store:** one SQLite file via aiosqlite plus `sqlite-vec` (`vec0` table `memory_vectors`). Tables: stories, story_versions, story_patches, story_sessions, conversation_turns, safety_events, memory_chunks, media_assets, child_interests. Rows are pydantic JSON in a `document` column.
- **Location:** `YARNOTRON_DATA_DIR`, else `XDG_DATA_HOME/Yarnotron`, else `~/Library/Application Support/Yarnotron` (macOS) / `%APPDATA%` (Windows) / `~/.local/share/Yarnotron`. Docker uses `/data`. Files: `yarnotron.db`, `generated_media/`, `preferences.json`.
- **Story:** id, title, synopsis, characters, scenes (ordered), facts, timestamps. Patch operations: update title/synopsis; add/update/remove character, scene, fact.
- **Versions:** every applied patch snapshots first; undo and `/versions/{id}/restore` use them.
- **Memory:** vec0 returns L2 distance; results are over-fetched, filtered by story and inverted to a score. With no embedding model, a deterministic hash pseudo-embedding is used.
- **Media:** child picture request → `pending_parent_review` → parent `approve`/`reject` → `generate` (placeholder SVG generator only). Statuses: pending_parent_review, approved, rejected, generated, failed.
- **Graceful shutdown** (`services/shutdown.py`): draining refuses new turns and sessions (503); waits for in-flight turns and background reviews up to a timeout, then cancels leftovers; ends open sessions; `PRAGMA wal_checkpoint(TRUNCATE)`; reports the story.

## 6. Backend HTTP API

Auth: if `YARNOTRON_API_TOKEN` is set, every `/v1/*` route needs `Authorization: Bearer <token>` (constant-time compare, 401 otherwise). Open: `GET /health`, `GET /dashboard` (HTML shell), `/media/*`.

| Method + path | Purpose |
|---|---|
| GET `/health` | `{status, app, environment}` |
| GET `/dashboard` | Parent web controls |
| GET `/v1/status?child_id=` | One cheap poll: `{app, version, environment, auth_required, strike_limit, stories, shutting_down, models{ollama_reachable, chat_model, safety_model, embedding_model, missing[]}, child{child_id, open_sessions, unreviewed_safety_events, pending_media, latest_session_id, latest_session_ended, latest_session_strikes}}`. Always 200 |
| GET `/v1/models/ollama` | Installed models; 503 if Ollama is down |
| POST/GET `/v1/stories`, GET/DELETE `/v1/stories/{id}` | Create (title 1–120, synopsis), list, read, delete |
| GET `/v1/stories/{id}/export.md` | Story as Markdown |
| POST `/v1/stories/{id}/patches` | Apply a patch |
| GET `/v1/stories/{id}/versions`, POST `…/versions/{vid}/restore` | Snapshots |
| POST `/v1/agent/turn` | `{child_id, story_id?, session_id?, device_id?, transcript(1–10000), model?}` → turn: `turn_id, session_id, intent, response_text, safety{action, severity, categories, child_message, parent_note}, story_patches, applied_story_patches, updated_story, tutor_notes, memory_chunk_ids, retrieved_memory_ids, media_asset_ids`. 503 if Ollama is down or draining |
| POST `/v1/agent/turn/stream` | Same turn as NDJSON (section 3) |
| GET `/v1/conversation-turns?limit=` | Recent turns (contains transcripts) |
| POST/GET `/v1/sessions`, POST `/v1/sessions/{id}/end` | Create (503 while draining), list, end |
| GET `/v1/safety-events?limit=` (1–250) | Events: `id, child_id, story_id, session_id, safety{…}, transcript_excerpt, counted_as_strike, reviewed_at, created_at` |
| POST `/v1/safety-events/{id}/review` | `{reviewed_by, review_note?}` |
| POST `/v1/safety-events/{id}/release` | `{released_by, review_note?}` (rules in section 4) |
| GET `/v1/media`, POST `/v1/media/requests`, POST `/v1/media/{id}/approve|reject|generate` | Picture workflow |
| POST `/v1/memory/search`, DELETE `/v1/memory/{id}` | Semantic memory |
| GET/PUT `/v1/preferences`, GET `/v1/preferences/voices`, POST `…/voices/sample` | Local voice preferences (play on the server host) |
| POST `/v1/system/prepare-shutdown?timeout=30` | Drain and save; returns `{draining, drained, waited_seconds, turns_in_flight_at_start, turns_still_running, background_tasks_finished, background_tasks_cancelled, sessions_ended[], story{story_id, title, scenes, characters, updated_at}, checkpoint, notes[]}` |
| POST `/v1/system/cancel-shutdown` | Resume taking turns |

## 7. Configuration (environment, prefix `YARNOTRON_`, also `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama |
| `DEFAULT_MODEL` | `gemma4:12b-mlx` | Story model |
| `SAFETY_MODEL` | `gemma4:e2b-mlx` | Safety reviewer |
| `EMBEDDING_MODEL` | `nomic-embed-text:v1.5` | Embeddings (unset → hash fallback) |
| `EMBEDDING_DIMENSIONS` | 768 | vec0 width |
| `REQUEST_TIMEOUT_SECONDS` | 30 | Ollama HTTP timeout |
| `MODEL_SEED` | none | Pin sampling for repeatable runs |
| `SAFETY_STRIKE_LIMIT` | 5 | Strikes before a session ends |
| `DATABASE_URL` | `<data dir>/yarnotron.db` | `sqlite:///path` or a path (Postgres rejected) |
| `MEDIA_OUTPUT_DIR`, `PREFERENCES_PATH` | under data dir | |
| `API_TOKEN` | none | Enables bearer auth (blank = none) |
| `APP_NAME`, `ENVIRONMENT` | `Yarnotron`, `local` | Labels |
| `YARNOTRON_DATA_DIR` | | Read directly, not a Settings field |

Host and port are uvicorn arguments, not settings. Model calls use `temperature 0`, `think false`, `num_predict 512` (160 for safety), JSON format.

## 8. Connectors

**Ollama** (native API): `GET /api/tags`; `POST /api/generate` (JSON format, streaming optional); `POST /api/embed`. Failures raise `OllamaUnavailableError`; malformed JSON is a subclass handled by the safety reviewer. Streaming extracts `response_text` as it is written, so speech starts before patches are finished.

**Speech (Home Assistant path).** The Assist pipeline owns wake word, STT and TTS. STT is Wyoming faster-whisper on the Mac (`10300`); TTS is Wyoming Piper (HA add-on). The backend never sees audio, only text.

**Speech (original local path, still in the repo).** `src/yarnotron/voice`: energy-threshold recorder, faster-whisper `base.en`, wake word by transcribing and regex-matching a phrase (default "interpreter"), TTS via Kokoro → macOS `say` (Piper opt-in), Silero-based barge-in. Launched with `scripts/start_yarnotron.py --voice`. **Known gap:** its `BackendClient` sends no bearer token, so it fails against a backend with `YARNOTRON_API_TOKEN` set.

**Home Assistant:** custom integration (section 9).

**Mac supervisor:** section 10.

**Deployment:** `scripts/start_yarnotron.py` (uvicorn, health wait, PID/log under `.yarnotron/`); Dockerfile (python:3.13-slim, no voice extras, non-root, `/data` volume, healthcheck) and compose (refuses to start without an API token; Ollama external). **The image has never been built.**

## 9. Home Assistant integration (`custom_components/yarnotron`)

Domain `yarnotron`, version 0.1.0, `config_flow`, `iot_class local_polling`, depends on `conversation`. Tested against Home Assistant 2026.2 only.

### 9.1 Config entry

| Key | Where | Notes |
|---|---|---|
| `url` | data | Normalised base URL |
| `api_token` | data | Optional |
| `child_id` | data | Default `local_child`; one entry per child; unique id `url|child_id` |
| `scan_interval` | options | Seconds, 10–600, default 30 |
| `idle_minutes` | options | 1–720, default 30; new session after idle |
| `keep_listening` | options | Default true |
| `include_transcripts` | options | Default false (see 9.7) |
| `voice_power` | options | Default true: allow voice launch/shutdown |
| `supervisor_url`, `supervisor_token`, `mac_address` | options | Mac control; MAC enables Wake-on-LAN |

Flows: user, reauth, reconfigure, options. Errors: `invalid_url`, `cannot_connect`, `invalid_auth`, `unsupported_backend`, `unknown`, `wrong_child`; options add `invalid_mac`, `supervisor_auth`, `supervisor_url_needed`, `supervisor_not_supervisor`. An unreachable supervisor is accepted when saving (the Mac may be asleep).

### 9.2 Entities

Device "Yarnotron" (service type). Unique id `<entry_id>_<key>`.

| Platform | Key / name | State and attributes |
|---|---|---|
| sensor | `unreviewed_safety_events` | count |
| sensor | `session_strikes` | count; attrs `strike_limit`, `session_ended` |
| sensor | `last_safety_event` | timestamp of newest non-allow event; attrs `event_id, action, severity, categories, reviewed` |
| sensor | `pending_images` | count awaiting approval |
| sensor | `stories` | count (diagnostic) |
| sensor | `mac_services` | enum `all_running / degraded / stopped / unreachable`; attrs `services{name:state}, problem, error` (needs supervisor) |
| binary_sensor | `model_server` | connectivity = Ollama reachable; attrs model names, `missing_models` |
| binary_sensor | `session_locked` | problem = locked; local state, always available |
| binary_sensor | `mac_supervisor` | connectivity; attrs `problem, error` (needs supervisor) |
| switch | `storytime` | restore-state; off → agent refuses to start stories |
| button | `start_new_session`, `end_session` | |
| button | `launch`, `shutdown` | needs supervisor; shutdown has no confirmation |
| event | `safety_alert` | types `caution`, `blocked`, `session_ended`; attrs `event_id, action, severity, categories, story_id, session_id, counted_as_strike, created_at` (+ `transcript`, `parent_note` only if `include_transcripts`) |
| conversation | the agent | streaming; English only |

Alert type: `session_ended` if categories include `strike_limit_reached` or `session_ended`; `blocked` if action is block/alert_parent; else `caution`.

### 9.3 Services (`yarnotron.*`)

All accept optional `config_entry_id` (required only with several entries).

| Service | Fields | Response |
|---|---|---|
| `review_safety_event` | `event_id`, `note?` | none; actor recorded as `ha:<user>` or `ha:automation` |
| `release_safety_event` | `event_id`, `note?` | none |
| `start_new_session`, `end_session` | | none |
| `set_story` | `story_id?` (empty clears) | none; ends the session, pins the next one |
| `list_stories` | | `{stories[]}` |
| `export_story` | `story_id` | `{story_id, markdown}` |
| `launch` | `wait` (true) | `{ok, kind, summary, details}`, or `{started, kind}` if `wait` false |
| `shutdown` | `force` (false), `wait` (true) | same shape |
| `status` | | `{ok, summary, components[], markdown}` |

`components[]`: `{name, side: home_assistant|mac, state: ok|down|problem|attention|unknown, detail, issue: {summary, detail, hints[]} | null}`.

### 9.4 Events and notifications

- Event bus `yarnotron_power_finished`: `{kind: launch|shutdown, ok, summary, entry_id}`.
- Persistent notification `yarnotron_power_<entry_id>` when a launch/shutdown needs attention (dismissed on success); `yarnotron_status_<entry_id>` when a status check finds problems.
- Blueprint `blueprints/automation/yarnotron/safety_alert_notify.yaml`: trigger on the `safety_alert` event entity; inputs `alert_entity`, `alert_types` (default blocked, session_ended), `notify_action`; queued mode.

### 9.5 The conversation agent

Order of handling for each utterance:

1. A pending shutdown confirmation (60 s window): "yes" confirms, "no" cancels, anything else cancels silently and is handled normally.
2. **Power phrases** (whole-utterance match; work even with Storytime off or the session locked): *launch/start/boot/wake up/turn on/power on/fire up/bring up Yarnotron*; *shut down/shutdown/turn off/power down/close/quit/exit Yarnotron*; *Yarnotron status / status / is Yarnotron running*. Status is always allowed; launch/shutdown need `voice_power`; shutdown asks "say yes, shut down" first. Voice waits up to 50 s for a result, then says it is still working.
3. Storytime off → fixed message, backend untouched.
4. `powered_down` (set after a successful shutdown, cleared by launch) → fixed message.
5. Local controls: help, repeat, sleep, stop (stop ends the session). Work while locked.
6. Story turn → streamed turn; each sentence goes into the chat log as it arrives; errors become a spoken apology; auth errors start re-auth.

### 9.6 Session lock and state

Stored under `yarnotron.<entry_id>.session`: `session_id, story_id, pinned_story_id, locked, last_activity`. A turn whose categories include `strike_limit_reached` or `session_ended` locks the session; while locked nothing reaches the backend. Only the "Start new session" button/service clears it. It persists across Home Assistant restarts and is reconciled with `/v1/status` on the first turn after a restart.

### 9.7 Polling and privacy

- Coordinator polls `/v1/status` and the last 25 `/v1/safety-events`; alerts are de-duplicated (last 200 ids kept). Events older than the window that arrived while HA was down are not replayed. The agent loads even if the backend is down.
- **Child's words stay out of entity state and attributes** unless `include_transcripts` is on. This includes the backend's `parent_note`, which can quote the child (a strike-limit note ends with up to 200 characters of what they said). That leak existed in the first release and is fixed on this branch, with a regression test.
- Diagnostics redact `api_token`, `supervisor_token`, `supervisor_url`, `mac_address`, `url`, `transcript`, `transcript_excerpt`, `parent_note`.
- "Repeat that" keeps only the last reply, in memory.

### 9.8 Power control from Home Assistant

`PowerController` builds a two-sided report:

- **Home Assistant side:** integration loaded; voice agent registered; backend link (last poll); Wyoming `stt`/`tts` entities; assist satellites; story session lock.
- **Mac side:** supervisor reachable; Ollama; Whisper; Yarnotron backend.

Launch: wake the Mac by Wake-on-LAN if the supervisor is unreachable and a MAC is set (waits up to 90 s), run the supervisor launch job, then clear a leftover "shutting down" flag on the backend. Shutdown: end the HA session, run the supervisor shutdown job (which asks the backend to drain and save first). If the supervisor is unreachable but the backend answers, it still asks the backend to save and reports that it could not stop it. One operation runs at a time; asking again joins it.

## 10. Mac supervisor (`src/yarnotron/supervisor`)

Stdlib-only HTTP service, default `0.0.0.0:8990`, run under launchd (`scripts/install_supervisor_launchd.sh`, wrapped in `caffeinate -i`). Config: `~/.yarnotron-supervisor/supervisor.toml`; logs in `<state_dir>/logs/<service>.log`.

| Endpoint | Purpose |
|---|---|
| GET `/v1/health` | open: `{ok, api: 1, auth_required}` |
| GET `/v1/status` | `{ok, services[{name, label, state: up|down|unresponsive, ok, detail, latency_ms, pids, process, log}], busy, running_job}` |
| POST `/v1/launch` | 202, a Job |
| POST `/v1/shutdown?force=1` | 202, a Job |
| GET `/v1/jobs/{id}` | Job `{id, kind, state: running|done, ok, summary, steps[], story, started_at, finished_at}` |

Step: `{service, label, outcome: already_up|started|stopped|already_stopped|left_running|failed, message, seconds, warnings[], issue, extra}`. Issue: `{service, code, summary, detail, hints[]}`.

Service config: `name, label, start[], port, cwd, env, depends_on, start_timeout, stop_timeout, stop_policy (term|leave), process_match, restart_unresponsive, probe{type: http|wyoming|tcp, …}, pre_stop{url, method, timeout, bearer_token}`. The example defines Ollama → Whisper → Yarnotron backend (depends on Ollama; `pre_stop` = `POST /v1/system/prepare-shutdown`).

Behaviour: launch in dependency order, probing for *answering* (not just listening); a hung process holding a port is restarted only if its command matches `process_match`, never otherwise. Shutdown in reverse order; the backend is not stopped if saving fails or work is unfinished unless forced; stop is SIGTERM then SIGKILL, and a service still answering afterwards is reported (something is restarting it).

Issue codes: `command_not_found, spawn_failed, exited_early, start_timeout, port_in_use, dependency_failed, prepare_failed, not_drained, kept_running, no_pid, still_running`. Each carries a spoken-length summary, the observed detail (including the log tail) and hints.

## 11. Hermes profile (this branch)

`integrations/hermes/`: a stdio MCP server (`yarnotron_mcp.py`), a profile persona (`SOUL.md`), a config snippet and a skill. It is a **parent/operator** agent; it has no tool that sends the child anything.

| MCP tool | Effect | Guard |
|---|---|---|
| `status` | Both-sided status with explanations | read |
| `launch` | Start what is not answering | safe to repeat |
| `shutdown` | Save + stop | `confirmed=true`; `force` needs `YARNOTRON_MCP_ALLOW_FORCE=1` |
| `safety_events` | Alerts without transcript or parent note | read |
| `review_event` | Mark reviewed (actor recorded) | `confirmed=true` |
| `release_event` | Release held-back content | off unless `ALLOW_RELEASE=1`; `confirmed=true` |
| `list_stories` | Titles and counts | read |
| `story_text` | Story Markdown | off unless `ALLOW_STORY_TEXT=1`; `confirmed=true` |
| `digest` | Counts for a scheduled summary | read |

When `HASS_URL` + `HASS_TOKEN` are set, status/launch/shutdown go through the `yarnotron.*` services so Home Assistant's own state stays right; otherwise the server calls the Mac supervisor directly. Hermes mounts the tools as `mcp_yarnotron_<tool>` (for example `mcp_yarnotron_status`); servers are listed under `mcp_servers` in the profile's `config.yaml` with `tools.include` as an allowlist, and Hermes does not pass the shell environment to MCP servers.

## 12. Boundaries for Jeevus

**Credentials in play**

| Credential | Held by | Grants |
|---|---|---|
| Backend `API_TOKEN` | HA entry, supervisor `pre_stop`, MCP server | All `/v1/*`, including the child's transcripts |
| Supervisor token | HA options, MCP fallback | Start/stop of Ollama, Whisper, backend |
| HA long-lived token | MCP server | Whatever that HA user can do |

**Suggested rules, stated as proposals for discussion**

1. Keep the child's voice path outside any general-purpose agent. Jeevus interacts with Yarnotron as an operator, not as a conversation partner.
2. Treat everything the child said or wrote as non-exportable by default. Counts, categories, actions and timings are the agreed currency.
3. One authority per concern: the supervisor starts/stops processes; Home Assistant owns the child's session lock and the voice pipeline; the backend owns safety and stories. Jeevus asks, it does not duplicate.
4. Destructive or releasing actions need a human yes in the same conversation, plus an allowlist on the tool side.
5. Treat story titles, error text and safety categories as untrusted data inside an agent's context.

**Failure modes worth designing for**

- The Mac sleeps (MacBook): nothing on it answers until woken; Wake-on-LAN over Wi-Fi is unreliable. The launchd agent runs only while the user is logged in.
- Ollama's menu-bar app can restart the server after the supervisor stops it (reported as `still_running`).
- A shutdown that cannot save leaves the backend draining (503 on new turns) until launch/`cancel-shutdown`.
- Home Assistant restarts: session lock persists; the alert window is 25 events.

## 13. Open questions for the Jeevus discussion

1. Is Hermes allowed to run on the same machine/account as the Yarnotron database? (Profiles do not sandbox the filesystem.)
2. Which Home Assistant user should Jeevus act as, and should it be limited to `yarnotron.*` services?
3. Should safety alerts also push to Jeevus (event-driven), or is polling by digest enough?
4. Does Jeevus need cross-agent memory of Yarnotron facts? If so, only counts and operational notes, never content.
5. Should Jeevus be able to change `voice_power`, `storytime` or schedules (bedtime hours), and with what approval?
6. One child or several? One backend entry per child today; the supervisor and backend are shared.
7. Where should the story model run long-term (see the hardware plans), and should Jeevus' LiteLLM routing serve the story and safety models, or stay separate so safety is not routed to a cloud model by accident?
8. Is it acceptable for text (not voice) to reach a cloud model in a hybrid setup? Earlier decision: yes for text, no for voice; the safety reviewer should stay local regardless.

## 14. Known gaps and verification state

**Run and passing (on Linux):** 313 backend tests (including 16 supervisor tests against real throwaway processes and the shutdown drain), 94 Home Assistant tests against Home Assistant 2026.2.3 with fake backend and supervisor over real HTTP, 15 Hermes MCP tests (real stdio protocol, stdlib fake servers). A real uvicorn backend was launched, had a session opened, was shut down and stopped by the supervisor.

**Not verified**

- Anything on real hardware: a voice satellite, the launchd install, `lsof` on macOS, `uv` start-up, Ollama's app behaviour, Wake-on-LAN.
- A live Hermes install; the MCP server was tested against the protocol, not Hermes itself. Hermes' skills folder for a named profile and its cron syntax are **unverified**.
- Home Assistant's `?return_response` payload on your version (tested against the documented shape).
- The Docker image (never built).
- Whether SQLite runs in WAL mode (the code only checkpoints; harmless if not).
- Whether deleting a story removes its rows from the `memory_vectors` virtual table (no foreign key; not checked).
- Dashboard token handling in the browser (not read).
- `docs/roadmap.md` plans (GUI, voice profiles, online deployment) were not reviewed for this document.

**Known limitations**

- The legacy local voice loop cannot use an API-token-protected backend.
- Home Assistant agent is English only; prompts and safety are English.
- Keyword guard and intent routing are substring/whole-word heuristics; the model reviewer is the second line and fails open if its output is invalid.
- Pictures are placeholder SVGs.

## 15. Repository map

| Path | Contents |
|---|---|
| `src/yarnotron/` | Backend: `agent/`, `api/`, `core/`, `models/`, `services/`, `voice/`, `supervisor/` |
| `custom_components/yarnotron/` | Home Assistant integration |
| `blueprints/automation/yarnotron/` | Safety alert automation |
| `integrations/hermes/` | MCP server, Hermes profile, tests |
| `scripts/` | Start/stop, supervisor runner and launchd installer, migration |
| `supervisor.example.toml` | Supervisor config example |
| `tests/`, `ha_tests/`, `integrations/hermes/tests/` | Backend, Home Assistant, Hermes tests |
| `docs/` | Architecture, phase notes, HA guide (HTML), Mac power control, this file |

Branches: `main` (Home Assistant integration), `feature/voice-power-control` (supervisor, shutdown, power control), `feature/hermes-profile` (this work, built on the previous branch).
