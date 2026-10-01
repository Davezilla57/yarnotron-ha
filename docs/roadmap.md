# Yarnotron Roadmap

## Near-Term Milestones

1. Replace the SVG placeholder generator with a local image backend such as ComfyUI, Flux, or Stable Diffusion.
2. Add parent-control authentication before exposing the dashboard beyond localhost.
3. Add a pending-edit confirmation queue so deferred destructive edits can be applied by a later "yes, confirm" turn.
4. Add parent dashboard visibility for stored child interests, including a way to remove one.

## Future Cross-Platform GUI

Build a GUI that runs on macOS, Linux, and Windows for both parent controls and local voice-session operation.

Candidate approaches:

- Tauri desktop app with a web frontend and native shell.
- Electron app if faster iteration and broader package ecosystem matter more than footprint.
- Local web app served by the FastAPI backend for the lowest operational complexity.

Desired GUI capabilities:

- Parent dashboard with stories, sessions, safety events, memory search, and story version rollback.
- Voice settings for microphone, speaker, STT model, TTS provider, and preferred voice.
- Child-facing session screen with large controls, mute state, transcript preview, and gentle status indicators.
- Story browser, export, and parent-approved generated media gallery.

## Future Voice-Activated Profiles

Add optional local speaker recognition so Yarnotron can select the active child profile from the child's voice after the wake word. Treat voice recognition as a convenience feature for profile selection, not as authentication or a parent-control security boundary.

Desired behavior:

- Parent creates or approves profiles and records a few short enrollment phrases for each child.
- The local voice loop identifies the likely speaker before sending the story turn to the backend.
- The active profile controls child id, age-aware safety settings, preferred voice, and the default story workspace.
- If confidence is low or multiple profiles match, Yarnotron asks a short confirmation question instead of guessing.
- Parent dashboard shows profile enrollment status, last-used profile, and a way to disable voice recognition.

Design constraints:

- Keep speaker embeddings and enrollment audio local by default.
- Avoid storing raw enrollment audio unless the parent explicitly enables it for troubleshooting.
- Use an open local speaker-recognition model so the feature can run on macOS now and later on Linux or Windows.
- Preserve manual profile selection as a fallback for shared microphones, noisy rooms, guests, or changed voices.

## Future Efficient Online Deployment

Make Yarnotron efficient enough to run as an online or always-on home application on modest hardware, such as a Raspberry Pi, mini PC, NAS, or small DigitalOcean droplet. The long-term goal is to keep the child-facing voice experience responsive while moving expensive work to optional workers or larger machines when needed.

Target deployment shapes:

- Home hub: Raspberry Pi or small x86 box runs FastAPI, the SQLite datastore, parent dashboard, wake/listen loop, and lightweight TTS/STT.
- Hybrid home setup: modest always-on hub handles voice I/O and story state while a Mac, gaming PC, or cloud worker handles heavier model calls and media generation.
- Small cloud app: low-cost droplet runs the backend, dashboard, datastore, and queues; voice clients connect from home devices.
- Split media pipeline: story turns stay low latency, while images and videos are queued, safety-reviewed, and generated asynchronously.

Steps to get there:

1. Add runtime profiling for STT, LLM calls, embeddings, database writes, TTS, media generation, and end-to-end voice latency.
2. Define hardware tiers with recommended model choices for MacBook Pro, RTX PC, mini PC, Raspberry Pi, and small cloud VM.
3. Add configuration profiles such as `local-dev`, `home-hub`, `hybrid-worker`, and `cloud-small`.
4. Containerize the backend, worker, database, and dashboard with production-oriented Docker Compose files.
5. Move slow or optional work into background jobs, starting with image/video generation and long-running story exports.
6. Add a worker interface so heavier machines can register as trusted local or remote model workers.
7. Support smaller model modes for low-power hardware, including lightweight embeddings, tiny/base STT, and compact story-agent models.
8. Add response caching and memory-search limits so repeated recaps, definitions, and story summaries avoid unnecessary model calls.
9. Add database maintenance tasks for memory chunk cleanup, old session archival, and generated-media retention.
10. Add deployment documentation for Raspberry Pi OS, Ubuntu server, DigitalOcean droplets, and a home-LAN-only setup.
11. Add observability suitable for parents and operators: health checks, service status, queue depth, disk usage, model latency, and microphone status.
12. Add secure remote access options before internet exposure, including parent authentication, HTTPS, backup/restore, rate limits, and audit logs.

Design constraints:

- Keep the voice path responsive by prioritizing wake word, recording, STT, safety review, story-agent response, and TTS over background work.
- Make online deployment optional; local-only operation should remain a first-class mode.
- Treat child data as private by default, with clear controls for what can leave the home network.
- Prefer graceful degradation over hard failure: if media generation or a remote worker is unavailable, story conversation should continue.
- Avoid assuming GPU access for the main application server.
