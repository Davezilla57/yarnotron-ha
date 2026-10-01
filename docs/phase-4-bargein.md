# Phase 4: barge-in, without the rewrite

Phase 4 as written replaces the Python voice client with three Rust crates and,
at the end, adds barge-in. The rewrite was deferred and barge-in was built in
Python instead. This records why, and what barge-in actually took.

## Why the rewrite was deferred

The teardown doc is candid that phases 3–7 "serve you rather than the kid at
the microphone". For phase 4 the split is sharper than that: **barge-in is the
only child-facing item in it, and it does not need Rust.**

| the claim | what was measured |
|---|---|
| Rust improves latency | Phase 0: all non-model work is under 40 ms. Client first audio is ~2.2 s, of which the backend and synthesis are essentially all of it. Rust moves neither. |
| barge-in needs the rewrite | No. `sd.play` returns in 20 ms, capture runs during playback, `sd.stop` cuts an utterance in 107 ms. |
| "match or beat phase 1–2 on the same harness" | That baseline did not exist. Phases 1 and 2 were measured on the backend and the synthesiser separately, never through the real loop with a real speaker. |

The rewrite is still worth doing for ownership and packaging. It is worth doing
*after* there is a working, measured reference implementation to port, rather
than before.

## The phase 1–2 client baseline, which did not exist before

`scripts/bench_voice_client.py` drives the real loop with a supplied transcript,
so it measures everything after speech recognition: backend stream, sentence
gating, synthesis and playback through actual speakers.

| | p50 | p95 |
|---|---:|---:|
| time to first audio | **2189 ms** | 2748 ms |
| whole turn, including speaking the reply | 9131 ms | 9132 ms |

This is what a Rust client has to match. It was better than predicted: phase 2
estimated first audio would land near 3020 ms once Kokoro's ~350 ms was added
to the backend's 2674 ms, and the real figure is ~2.2 s.

## Barge-in

### The problem is not the plumbing

Listening while speaking is easy. Telling the child apart from Yarnotron is
not. Measured on MacBook speakers:

| | rms p50 | frames Silero calls speech |
|---|---:|---:|
| room, nothing playing | 0.00108 | 0% |
| room, our own reply playing | 0.03275 | **58%** |

Our own reply raises the microphone's noise floor **30x**, and the VAD calls it
speech more than half the time — correctly, because it *is* speech. Energy does
not separate them. A speech classifier does not separate them.

### What does separate them

The child is nearer the microphone than the speakers are. So the detector
calibrates against our own echo instead of assuming silence:

1. a grace window at the start of each utterance, where the echo level is
   measured and nothing can trigger — this also swallows the tail of the
   previous utterance,
2. a threshold set from that measurement, so it follows the volume and the room
   rather than being a constant that goes stale,
3. sustained speech above it, so a cough or a chair scrape is not an
   interruption.

### The constants were tuned, not guessed

The first attempt used echo p90 × 1.8 and nine consecutive frames. It never
fired, even on a child at twice normal volume. Two reasons, both visible only
once measured against a real recording:

- the threshold landed at 0.176 while the child's frames peaked around 0.20, so
  almost nothing cleared it,
- speech fluctuates, so nine *consecutive* frames above a threshold essentially
  never happens.

Sweeping percentile, margin and window against recorded echo:

| | result |
|---|---|
| p95 × 1.1, 6 of last 12 frames | silent on echo, fires at 1.28 s on a child starting at 1.0 s |
| p95 × 1.3, 8 of 16 | misses a normal-volume child |
| p95 × 1.6, any window | misses everything but a shout |

**Detection latency is 280–400 ms** after the child starts talking.

### An honest limitation

A quiet or distant child is not detected. The gap is genuinely narrow — echo
peaks around 0.10 rms while a child near the microphone sits around 0.11
median — and pushing the threshold lower starts triggering on our own voice.

The real fix is acoustic echo cancellation, which subtracts the known playback
signal from the captured one. That wants the reference signal plumbed through
the audio path and is a bigger piece of work. Headphones make the problem
disappear entirely.

## What shipped

- `voice/bargein.py`: `BargeInDetector`, calibrating and sustained-speech gated
- `KokoroSpeaker.speak_until_interrupted`: non-blocking playback with the
  microphone monitored throughout
- the loop stops the reply where the child cut in and drains the rest of the
  stream unspoken, so the turn is still recorded and persisted
- `scripts/bench_voice_client.py`: the client baseline that did not exist
- 12 detector tests on synthetic signals, so they need no audio hardware, and 4
  loop tests for the interruption path

Speakers without `speak_until_interrupted` — macOS `say`, Piper — are
unaffected and simply cannot be interrupted.

Barge-in costs nothing when nobody interrupts: first audio measured 2189 ms
with it enabled against 2220 ms without, which is noise.

## What the rewrite still owes

When Rust does happen, it inherits a specification rather than a guess:

- the client baseline to beat, 2189 ms p50 to first audio
- a detector design with tuned constants and the measurements behind them
- the knowledge that `format: <schema>`, `seed` on non-streaming requests, and
  naive VAD barge-in all do not work here

The Rust pipeline's real advantage is structural: capture as a permanent tokio
task rather than a stream opened per utterance, which removes the grace-window
hack and makes proper echo cancellation somewhere to put.
