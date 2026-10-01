# Phase 1: stream everything

Phase 1 of the architecture teardown. The goal was 6.7 s to roughly 1.2 s
without changing the language, the database or the packaging.

**Outcome: first audio 5880 ms → 2806 ms, a 2.1x improvement. The 1.5 s exit
criterion was not met, and the measurements below show it is not reachable
while every spoken sentence is gated by a model.**

## What shipped

| item | status |
|---|---|
| streaming Ollama client + incremental JSON reader | done |
| streaming turn endpoint, buffered one kept | done |
| sentence-by-sentence delivery to TTS | done |
| per-sentence output safety gate | done |
| patch application and memory writes after first audio | done |
| input safety concurrent with generation | **not done** — see below |
| Silero VAD, endpoint tail 900 ms → 250 ms | **not done** — see below |
| preload and pin both models | **skipped** — measured as a no-op |
| stabilise the prompt prefix | **skipped** — already effective |

## Latency

Full 56-turn corpus, `gemma4:e2b-mlx` safety, seed 42, both runs back to back
on a machine with nothing else holding the GPU. 56/56 completed in both.

| | buffered | streamed |
|---|---:|---:|
| **first audio** p50 | — | **2806 ms** |
| total turn p50 | 5880 ms | 6473 ms |

**First audio 5880 → 2806 ms, 2.1x.**

Note the second row: streaming makes the *total* turn about 600 ms slower,
because three per-sentence safety gates cost more than one review of the whole
reply. That is the right trade for a voice app — the child hears a reply in
half the time — but it is a trade, and it will matter to anything that consumes
the buffered endpoint.

Where the 2806 ms goes:

| segment | ms | can streaming help? |
|---|---:|---|
| input safety | 334 | no — runs before generation starts |
| prompt eval | 1376 | no — the model must read the prompt first |
| generate first sentence | ~700 | no — JSON preamble plus ~15 tokens |
| output safety gate | 383 | no — the sentence must pass before it is spoken |

Prompt evaluation is now the single largest component. It grew from the 582 ms
measured on a 6-turn sample because that sample only covered early turns, when
the story is small; over a full session the inlined story document gets
substantially longer. **That makes phase 5's story compiler a first-audio
optimisation, not just a prompt-cost one** — contrary to what the phase 0
baseline concluded from its flat drift table.

**Nothing left on that list is streaming-shaped.** The 5 s that streaming
removed was the wait for the *rest* of the draft — the patches, the tutor
notes, the closing brace. That is gone. What remains is the irreducible cost of
reading a prompt, generating a sentence, and checking it twice.

### Why 1.5 s is not reachable as specified

Making input safety concurrent with generation is the only remaining item on
the phase 1 list that removes time from the critical path, and it lands at
roughly 2470 ms — still well over. Going below 1.5 s requires one of:

- **Shorten the prompt** (−1376 ms at most, the largest single lever). This is
  phase 5's story compiler. It was pulled forward and done: about 560 ms per
  turn on a sixteen-scene story, more as the story grows. See
  `docs/phase-5-story-compiler.md`.
- **Drop the per-sentence model gate from the critical path** (−383 ms). The
  first sentence would be spoken having passed only the keyword guard, with the
  model review running alongside and cutting playback short on a veto. That is
  a real safety change: unsafe wording could reach the child for one sentence.
- **A faster safety model.** `gemma4:e2b-mlx` at 346 ms is the fastest
  candidate that actually catches hazards; `qwen3.5:0.8b-mlx` is 149 ms and
  allows everything (see `docs/phase-0-baseline.md`).

The plan's own arithmetic assumed a 0.8B gate at 149 ms. That model is
unusable, and its replacement costs 197 ms more per gate.

## Two items skipped, with evidence

### Silero VAD and the 900 ms endpoint tail

The plan bundles "replace RMS VAD with Silero" and "drop the endpoint tail to
~250 ms" as one action. They are independent, and only one of them is safe.

`silence_seconds` is a plain constant in `voice/audio.py`. The tail can be
shortened today with no Silero and no new dependency. Silero itself is also
nearly free: `faster-whisper` already ships Silero VAD v6 as a 1.2 MB ONNX
model and `onnxruntime` is already installed, running at 0.06 ms per 32 ms
frame. (The `silero-vad` PyPI package pulls in torch — do not use it.)

What the tail actually costs, on synthesised utterances with a realistic noise
floor. "CUT EARLY" means the speaker was cut off mid-utterance:

| scenario | tail | RMS | Silero | Silero + LSTM state |
|---|---|---|---|---|
| 600 ms thinking pause | 0.90 | ok | ok | ok |
| | 0.50 | CUT | CUT | CUT |
| | 0.35 | CUT | CUT | CUT |
| trails off quietly | 0.35 | ok | ok | ok |
| | 0.25 | CUT | CUT | CUT |
| clean utterance | 0.25 | ok | CUT | ok |

At 250 ms everything breaks, and Silero does not rescue it. A 600 ms
mid-sentence pause needs a ≥900 ms tail whichever algorithm is used — no VAD
can know the child is about to resume. Silero's genuine edge is narrow: it
tracks trailing-off quiet speech about 280 ms longer, and with recurrent state
carried across frames it avoids a regression the stateless wrapper shows.

Skipped for this phase. Phase 4 adds barge-in, which is what actually makes a
short tail safe: a child who gets cut off can simply keep talking.

### Model pinning and prompt-prefix stability

Both measured as no-ops and were not implemented.

Prompt-prefix caching already works: repeated calls with the same prefix report
`prompt_eval_duration` of 0 ms, because the static instruction block sits at
the front of the prompt and the variable story state at the end. Nothing to
stabilise.

Model pinning only affects the first turn after a cold start. At p50 it is 0 ms.
It is ~10 lines in the lifespan if the first turn of a session ever matters
enough.

## Only the streaming path is reproducible

Verifying "streaming did not change behaviour" turned up something bigger, and
took two attempts to read correctly.

Two **buffered** runs of the same 12 turns, back to back with identical
configuration, disagreed on 25% of patch decisions and 67% of wording. The
first conclusion — that `temperature: 0` does not make `gemma4:12b-mlx` repeat
itself — was half right and pointed at the wrong culprit. Adding a seed did not
fully fix it either, which is what prompted a closer look.

Same prompt, same seed, eight prompts, three runs each:

| | reproduces itself |
|---|---|
| `stream: false` (buffered) | **1/8** |
| `stream: true` (streamed) | **8/8** |
| buffered output matches streamed output | 1/8 |

**Ollama's MLX runner does not honour `seed` when `stream` is false.** With
streaming on it is exact. The model was never the problem; the non-streaming
request path was.

That alone does not make the corpus reproducible, though. Two **streamed**
runs, same seed, full corpus, back to back still differ:

| field | identical |
|---|---|
| intent | 56/56 |
| safety action | 54/56 |
| patch shape | 40/56 |
| response text | 14/56 |

The reason is visible in the very first turn. t01's two replies are identical
for **138 characters** and then fork — "making that noise `outside the door?`"
against "`behind the heavy door?`". Same prompt, same seed, same path. That is
floating-point nondeterminism in parallel reductions flipping one token late in
a generation, which is why the eight-prompt probe above saw 8/8: those
generations were shorter.

Then it cascades. The corpus is sequential: t01's reply becomes a memory chunk
and its patches change the story, so from t02 onward the prompts genuinely
differ. First text divergence is t01, first *decision* divergence is t03, and
by the end 42/56 turns differ in text and 16/56 in decisions. Those 16 are
mostly one early fork propagating, not 16 independent faults.

**So a whole-corpus replay cannot be a clean regression signal, seeded or not.**
That is a limitation of the harness design, not of streaming.

### What phase 5 needs instead

Replay each turn against **fixed, recorded story state** rather than letting the
corpus evolve. Each turn then becomes an independent comparison with a
byte-identical prompt, where a divergence means something and cannot cascade
into the next turn. The `bench/golden` file already records enough state to do
this; the harness just needs to stop threading one story through all 56 turns.

Until that exists, use the corpus the way `bench_safety_models.py` already does
— as a source of realistic payloads to score a component against — not as a
byte-exact oracle.

`YARNOTRON_MODEL_SEED` is unset by default, so a child does not hear identical
wording every session. `bench_turns.py --seed` sets it, `--stream` is required
for it to have any effect, and `scripts/diff_golden.py` reports decisions and
prose in separate columns.

## Did streaming change behaviour?

No, on the evidence available — but the exit criterion as written ("golden
corpus responses unchanged in substance") cannot be checked the way it was
meant to be, because of the reproducibility limits above.

What can be said:

- **Intent is 56/56 identical** in every comparison run: streamed against
  buffered, and streamed against streamed. Routing is unaffected.
- Streamed against buffered drifts *no more* than streamed against itself
  (patch shape 44/56 versus 40/56). If streaming had changed behaviour, the
  cross-mode comparison would be the worse of the two. It is not.
- Both paths fail and succeed on exactly the same turns, including the t16
  validation bug below, which surfaced identically in both.
- At the unit level the two paths are asserted to produce identical responses
  from identical input (`test_streaming_and_buffered_produce_the_same_response`),
  where the model is a fake and nondeterminism is removed.

The unit test is the load-bearing evidence. The corpus comparison supports it
but cannot prove it on its own.

## A bug the seed found: one bad patch destroyed the whole turn

With seed 42 pinned, both the buffered and streamed corpus runs failed on the
same turn, t16, with the same error:

```
ValidationError: 1 validation error for AgentDraft
story_patches.0.operation
```

gemma4 emitted a patch whose `operation` was not in `StoryPatchOperation`.
Because the whole draft was validated in one call, that single bad patch
discarded the entire turn — including a reply that was perfectly good. On the
streaming path it is worse: the reply has already been spoken, so the child
hears it and then gets an error.

Patches and tutor notes are now validated individually. A malformed one is
dropped and counted in `dropped_invalid_draft_items`; the reply and the valid
patches survive. `AgentDraft.response_text` also no longer requires a non-empty
string, which makes the `_fallback_response` that already existed downstream
reachable instead of dead code.

This was only findable because the seed made it reproducible. Unseeded it is an
occasional unexplained 500 in a child's session.

## Safety

Each sentence passes a model review before it is spoken, and a veto stops the
entire remaining reply rather than one sentence. The gate sees sentence text
only — the story patches have not been generated yet — so it does not replace
the full review over text *and* patches, which still runs before anything is
applied or persisted.

That preserves the property the architecture doc calls out as worth keeping:
the output reviewer sees the proposed patches. The gate protects what is
spoken; the full review protects what is kept.

## What phase 2 inherits

- First audio is 2806 ms, of which 1376 ms is prompt evaluation. Phase 2's
  Kokoro work adds synthesis time on top of that number, not instead of it.
- Prompt evaluation being the largest component means **phase 5's story
  compiler is worth pulling forward**. It is the only lever that moves first
  audio meaningfully without touching the safety design.
- The `field_done` signal matters more than it looks. Without it the last
  sentence waits for the story patches to finish generating — a measured 3 s
  silence gap mid-reply. Any reimplementation needs the equivalent.
- Every gemma draft arrives inside a ```json fence, so an incremental reader
  has to handle the fence from the first token, not as a fallback.
