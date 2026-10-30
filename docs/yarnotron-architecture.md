# Yarnotron Architecture

## Recommendation

Build Yarnotron as a local-first voice agent with optional Alexa integration later. Alexa can be useful as a familiar launch surface, but it is not the best primary interface for this project because a rich storytelling companion needs low-latency streaming audio, long sessions, local memory, tool calls, parent controls, and safety review around generated media.

The best first interface is a local microphone/speaker device on the home network, connected to the always-on Mac Studio hub that runs the AI services (the RTX PC and the MacBook Pro are optional accelerators). A good prototype target is Home Assistant Voice Preview Edition or a Raspberry Pi/ESP32-S3 voice satellite with a USB speaker/mic. The agent core should not depend on the device; devices should just stream audio in and play audio out.

## Target Experience

The child can say things like:

- "Let's keep working on my dragon story."
- "Change the villain into a misunderstood inventor."
- "Remove the scary part with the cave."
- "What does mysterious mean?"
- "Can you help me make that sentence sound better?"
- "Make a picture of the clock tower scene."

The agent should remember stories as editable artifacts, not just chat history. Every session should update canonical story state: title, premise, characters, places, scenes, timeline, facts, drafts, revisions, and parent-visible safety events.

## Hardware Options

### Best Prototype

Use the always-on Mac Studio hub as the local AI server; the RTX PC and MacBook Pro can add capacity. Use a small voice satellite for room audio.

Recommended first device:

- Home Assistant Voice Preview Edition
- USB-C powered
- dual microphones
- internal speaker for voice feedback
- 3.5 mm audio out for a better speaker
- physical microphone mute switch
- ESPHome/Home Assistant integration

This is the most practical low-friction path because it is already built for local/private voice workflows and can connect to local LLMs.

### Better Audio, Still Simple

Use a Raspberry Pi 5 or small mini-PC with:

- ReSpeaker USB Mic Array or similar far-field mic
- small powered speaker
- Wyoming Satellite, Rhasspy-style audio pipeline, or a custom WebRTC client

This gives more control over audio streaming and wake words, but requires more assembly.

### Alexa Skill

Treat Alexa as phase two, not phase one. A private Alexa skill can forward recognized utterances to your backend, but Alexa skills are more constrained around conversational flow, certification, child-directed policies, data collection, account linking, and raw audio access. It is better for "open Yarnotron" and simple command-style interactions than for the main creative session.

## Local Runtime

(Early plan; the reference deployment now runs everything on an always-on Mac Studio hub, with the PC and MacBook Pro as optional accelerators.) Use the RTX 4080 PC as the main development runtime first. It will give the best local latency and headroom for speech, LLM, embeddings, safety classifiers, and image generation. The MacBook Pro should also be able to run a strong local prototype, especially with MLX-compatible models, but the 4080 is the better initial AI server.

Target services:

- Wake word: openWakeWord, microWakeWord, or device-native wake word
- Voice activity detection: Silero VAD
- Speech-to-text: faster-whisper, whisper.cpp, or Home Assistant Whisper
- LLM: Qwen3 8B/14B, Gemma 3 12B/27B, or a similar current local instruct model
- Fast routing/classification: a smaller 1B-4B model
- Text-to-speech: Piper for low-latency local TTS, later upgrade to a more natural local voice
- Embeddings: bge-small/en, nomic-embed-text, or another local embedding model
- Vector store: PostgreSQL with pgvector
- Relational store: PostgreSQL
- Media generation: local Stable Diffusion/Flux-style image pipeline on demand, disabled by default without safety checks

## Service Layout

```text
voice device
  -> audio gateway
  -> VAD / streaming STT
  -> turn router
  -> safety pre-check
  -> story agent
  -> memory writer
  -> safety post-check
  -> TTS
  -> voice device

story agent
  -> relational DB
  -> vector search
  -> story state compiler
  -> grammar/definition tutor
  -> image/video job queue
  -> parent alert service
```

## Current Local Voice Prototype

The local prototype is a microphone-first CLI loop in
`scripts/local_voice_loop.py`. It records one utterance at a time through
`sounddevice`, uses RMS voice activity detection, transcribes the resulting WAV
with `faster-whisper`, sends the transcript to the existing FastAPI
`POST /v1/agent/turn` contract, then speaks `response_text` through a pluggable
speaker.

The audio layer is intentionally modular:

- `VoiceActivityRecorder` owns local microphone capture.
- `FasterWhisperTranscriber` owns local speech-to-text.
- `BackendClient` owns the HTTP backend contract.
- `Speaker` implementations own text-to-speech playback.
- `LocalVoiceLoop` composes those pieces and can later accept a dedicated device
  recorder or network audio gateway without changing the backend API.

The default TTS provider is `auto`, which uses macOS `say` on this machine and
falls back to printed responses if no local speaker provider is available. Piper
is already available as the first dedicated local TTS provider when a voice model
is supplied.

## Memory Model

Store both structured story state and conversational evidence.

Core tables:

- `children`: local profile, age band, parent settings
- `stories`: title, synopsis, status, current draft pointer
- `story_versions`: immutable story snapshots
- `characters`: name, traits, role, relationships
- `places`: name, description, story role
- `scenes`: sequence, summary, draft text, status
- `facts`: canonical continuity facts
- `events`: timeline events inside the story world
- `conversation_turns`: transcript, intent, safety labels
- `memory_chunks`: embedded searchable chunks linked to story entities
- `revision_requests`: add/remove/change commands and whether applied
- `media_assets`: generated images/video, prompts, safety review result
- `safety_events`: category, severity, transcript excerpt, parent alert status

The key design choice: the LLM should propose patches to story state, not rewrite memory directly. The backend validates and applies those patches.

Example patch:

```json
{
  "operation": "update_character",
  "story_id": "story_clock_tower_mystery",
  "character": "Milo",
  "changes": {
    "role": "inventor",
    "motivation": "wants to fix the town clock"
  },
  "child_confirmation_required": true
}
```

## Agent Behavior

Use several small agents/functions instead of one large prompt.

- `turn_router`: decides whether this is storytelling, editing, tutoring, safety, parent command, or media generation
- `story_compiler`: retrieves relevant story state and turns it into a compact context
- `story_editor`: creates structured patches for add/remove/change requests
- `narration_partner`: asks gentle questions and helps continue the story
- `language_tutor`: gives short grammar, spelling, or vocabulary help only when useful
- `safety_guard`: detects unsafe topics, redirects gently, and logs parent alerts
- `media_prompt_builder`: creates safe image prompts from approved story excerpts

For a 12-year-old, the agent should avoid constant correction. Prefer lightweight coaching:

- "That sentence works. A slightly smoother version would be..."
- "Mysterious means something is hard to explain or not fully known yet."
- "Do you want that to sound more funny, dramatic, or simple?"

## Safety Design

Use layered safety rather than relying on one LLM instruction.

Pre-response checks:

- classify the child transcript
- detect requests involving sexual content, self-harm, graphic violence, weapons, illegal behavior, personal data sharing, bullying, hate, or adult themes
- block media generation for risky topics
- redirect in a calm, non-shaming voice

Post-response checks:

- classify the assistant response
- classify generated media prompts
- classify generated images before showing them
- log safety labels and blocked outputs

Parent alert levels:

- `info`: mild boundary nudge, no alert by default
- `review`: concerning but not urgent, visible in parent dashboard
- `alert`: notify parent immediately
- `urgent`: stop session and instruct child to get an adult

For a child product, do not make "alert parent" a hidden surprise. The agent can say: "I can't help make that kind of story. I'm going to save a note for your parent because this is something an adult should know about."

## Media Generation

Start with still images only. Short videos can come later because they are slower, more expensive, and much harder to moderate.

Flow:

1. Child asks for an image.
2. Agent summarizes the scene.
3. Safety guard checks the request.
4. Agent asks for confirmation if needed.
5. Image generator runs asynchronously.
6. Image safety classifier checks the result.
7. Approved image appears in a parent-visible gallery.

Keep image generation off the low-latency voice path. The agent should say something like, "I'll make that while we keep writing."

## MVP Build Plan

### Phase 1: Local Text Story Memory

- FastAPI backend
- PostgreSQL + pgvector
- story CRUD
- structured story patching
- transcript ingestion
- simple web parent dashboard

### Phase 2: Local Voice Loop

- microphone input
- VAD
- streaming STT
- local LLM endpoint
- Piper TTS
- wake word
- low-latency turn-taking

### Phase 3: Child-Safe Story Agent

- safety classifier
- grammar and vocabulary tutor
- parent alert log
- editable story state
- story version history

### Phase 4: Dedicated Voice Device

- Home Assistant Voice Preview Edition or Raspberry Pi satellite
- local network audio gateway
- physical mute
- parent-controlled operating hours

### Phase 5: Images

- local image model on RTX 4080
- prompt safety
- generated image safety
- parent gallery

### Phase 6: Alexa Adapter

- private skill for simple commands
- account linking only if required
- no sensitive child memory stored in Amazon
- backend remains source of truth

## Initial Stack

- Backend: Python, FastAPI, Pydantic
- Agent orchestration: explicit service classes and typed tool calls
- Database: PostgreSQL + pgvector
- Queue: Redis/RQ or Dramatiq
- STT: faster-whisper or whisper.cpp
- TTS: Piper
- LLM serving on RTX 4080: vLLM, llama.cpp, Ollama, or LM Studio for early testing
- LLM serving on Mac: MLX-LM or llama.cpp
- Frontend: simple local web dashboard
- Deployment: Docker Compose

## Design Principles

- Local-first and parent-owned data
- Child memory is editable, inspectable, and deletable
- Store stories as structured artifacts, not only chat logs
- Keep voice latency under two seconds for normal turns
- Use small models for routing and safety where possible
- Generate media asynchronously
- Make safety boundaries gentle, explicit, and logged
- Keep Alexa optional
