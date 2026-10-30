# Changelog

## 0.2.0 - 2026-10-29

Host-neutral release for the always-on Mac Studio deployment (the backend in a container; Whisper and text-to-speech on the same machine).

### Home Assistant integration
- **Finished-speaking wait.** New options let you pick voice satellites' "Finished speaking detection" selects and set them to Relaxed (about 1.25 s) so a child who pauses is not cut off. Applied when Home Assistant starts.
- **Voice launch and shutdown are off by default** and the host supervisor is optional. With no supervisor the status report no longer lists a missing "Mac" component, and all wording says "host computer" instead of "Mac". Existing supervisor setups keep working.
- Stop phrases: "stop Yarnotron" (including misheard spellings), "stop the story", "end the story" end the session; "bye", "bye bye", "see you later" put it to sleep.

### Backend (agent engine)
- Talker/Scribe agent engine with code-enforced safety; per-turn confirmation only for removals, renames and clearing wording.
- Name matching for misheard character names; removals staged by code with a named question.
- Listening mode ("let me tell you"), held unfinished sentences, frustration repair, new/delete story handling, "Dad" requests.
- Idle-loop guard: a session pauses after three filler transcripts in a row (speech recognition inventing "Thank you.").
- "Mrs." no longer breaks sentences; background-sound tags are ignored; repeated questions are avoided.
- Per-session summary at `GET /v1/sessions/{id}/summary`; timing records kept on the data volume.

### Docs
- `docs/yarnotron-for-jeevus.md`, the architecture notes, the HA guide and the power-control guide now describe the Studio deployment and mark the supervisor as optional.
