# Phase 2: replace the voice

Move from an obviously synthetic system voice to one a child wants to keep
listening to, without giving back the latency phase 1 bought.

## What shipped

| item | status |
|---|---|
| Kokoro behind the `Speaker` protocol | done |
| tiered fallback Kokoro → Piper → system | done |
| voice picker in the parent dashboard, sample button working | done |
| tighter spoken style contract | done |
| A/B across candidate voices | done — `af_heart` chosen |
| distinct voices for narration and characters | not done (optional) |

## Kokoro costs about 350 ms, and only on the first sentence

Measured per sentence with `scripts/bench_tts.py`:

| | p50 | p95 |
|---|---:|---:|
| kokoro-onnx `af_heart` | **348 ms** | 452 ms |
| kokoro `create_stream`, first chunk | 340 ms | 461 ms |
| model load | 434 ms, once at start-up | |

The number that matters is the real-time factor: **0.12**. Synthesis is eight
times faster than playback, so while the child listens to a ~2.1 s sentence the
next one is ready in ~350 ms with over a second to spare. **Only the first
sentence's synthesis sits on the critical path.** First audio goes from 2674 ms
to roughly 3020 ms; every later sentence is hidden behind the one playing.

`create_stream` is not worth using. It saves 8 ms, because a sentence is
already about one chunk. Per sentence is the streaming unit, which is what the
voice loop does already, so nothing structural had to change.

`kokoro-onnx` is the right package: six dependencies and no torch, reusing the
`onnxruntime` that `faster-whisper` already installs. The torch-based `kokoro`
package pulls transformers and spacy to the same end.

### Not measured

Whether macOS `say` starts playback immediately. `say` blocks on an audio
output device that a background process cannot open, so the honest comparison
needs a foreground session. `bench_tts.py`'s `say` row times synthesis to a
file, which is not how the voice loop invokes it, and is labelled as such. The
+350 ms figure above therefore assumes `say` is effectively instant, which is
the least favourable assumption for Kokoro.

## Replies were far too long

A reply spoken aloud is time the child spends listening instead of telling the
story. Measured by synthesising every corpus reply (`bench_reply_length.py`):

| | before | after |
|---|---:|---:|
| mean | 9.64 s | **6.44 s** |
| p50 | 9.17 s | 6.19 s |
| p95 | 15.42 s | 8.77 s |
| max | **18.60 s** | **9.05 s** |

A third shorter on average, and the worst case halved: no child now waits
through an eighteen-second monologue to get a word in.

The phase 2 target of "average reply under 5 s" is not met at 6.44 s, and has
been relaxed rather than chased further. Pushing the prompt harder trades
against the warmth that is the point of the product.

Two things caused the original length, and the second was ours:

**The prompt's examples contradicted its instruction.** It asked for "1-3 short
sentences" while every worked example was three sentences and 120–190
characters. Examples teach length more strongly than instructions do. They are
now two sentences ending on the question, with a counter-example of what is too
long, and a test asserts every example stays under 110 characters so this
cannot quietly regress.

**Our own boilerplate was longer than a reply should be.** The confirmation
prompt read "That is a big story change, so I want to check before I touch it.
Do you still want me to make that change?" — 107 characters, five seconds of
speech, spoken on every destructive edit. It is now "That's a big change.
Should I make it?" The story intros were trimmed the same way.

### Measure speech length, do not estimate it

Earlier numbers in this work estimated speech time from a chars-per-second
ratio taken from a single sample. That ratio read 20% short — 0.0463 against a
real 0.0559 — which was enough to make a miss look like a pass. Reply length is
now measured by synthesising the text.

## Choosing the voice

`scripts/render_voice_samples.py` renders the same replies in every candidate
voice as WAV files, named `reply<N>__<voice>.wav` so the same line sorts
together across voices. Replies come from the golden corpus rather than being
invented, spread across the length range, and one of them carries the appended
confirmation prompt so that is audible too. Writing files rather than playing
them also sidesteps the audio device, which a background process cannot open.

**`af_heart` is the default**, chosen by ear across eight candidates.

The rendering turned up something worth more than the preference. Voices speak
at very different rates, on identical text:

| voice | mean audio | vs macOS `say` |
|---|---:|---:|
| bf_emma | 5.53 s | −3% |
| say-Samantha | 5.70 s | — |
| **af_heart** | **5.83 s** | **+2%** |
| am_fenrir | 5.99 s | +5% |
| af_bella | 6.08 s | +7% |
| bm_george | 6.52 s | +14% |
| am_michael | 6.53 s | +15% |
| af_nicole | 9.31 s | +64% |

Picking `af_nicole` over `bf_emma` would make every reply 68% longer to listen
to — more listening time than the entire style-contract exercise saved. **Voice
choice is partly a pacing decision, not only an aesthetic one.** `af_heart`
sits near the fast end, so the preference and the pacing agree here; they would
not have for several of the others.

## Everything else

54 Kokoro voices are read from the model itself rather than hardcoded, so the
dashboard picker stays correct if the model changes. Voice ids are expanded
into readable names — `af_heart` shows as "Heart (female), American English".
The sample button previews through the same engine the child will hear, so the
parent is not auditioning a voice the loop will not use.

`FallbackSpeaker` marks a failed engine dead for the session rather than
retrying it on every sentence, and goes silent rather than raising if
everything fails — a dead synthesiser should not end the story.
