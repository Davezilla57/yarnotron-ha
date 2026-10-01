# Phase 0: the pinned baseline

Recorded 2026-08-22 on Apple Silicon, Ollama with MLX models, Postgres 16 +
pgvector in Docker. This is the reference every later phase is measured
against. Regenerate with:

```sh
docker compose -p yarnotron up -d
uv run python scripts/bench_turns.py --safety-model gemma4:12b-mlx --seed 42 --write-golden
uv run python scripts/bench_stt.py
```

## Configuration measured

| | |
|---|---|
| story model | `gemma4:12b-mlx` |
| safety model | `gemma4:12b-mlx` (see finding 1 — this is today's real behaviour) |
| embedding model | `nomic-embed-text:v1.5` |
| STT | `faster-whisper` `base.en`, cpu/int8 |
| corpus | `bench/corpus.json`, 56 turns over one fixed story |
| result | 56/56 turns completed, 0 failures, all intents as expected |

> **Correction from phase 1.** This baseline was recorded buffered and without
> a pinned seed. Ollama's MLX runner ignores `seed` unless `stream` is true, so
> a buffered run cannot be reproduced at all — the golden corpus recorded here
> could not have detected a regression. It has been re-recorded with
> `--stream --seed 42`. Even then a whole-corpus replay is not a byte-exact
> oracle, because the corpus threads one story through all 56 turns and a single
> divergence cascades. See `docs/phase-1-streaming.md`. The latency numbers below
> are unaffected; only the corpus's use as a correctness oracle is.
>
> The absolute latencies here were also measured while another application held
> a model on the GPU (finding 2). Phase 1 re-measured on a clean machine: the
> buffered baseline is **5880 ms** p50, not 7452 ms.

## Turn latency

56 turns, one pass. `n` is lower than 56 for stages only the storytelling path
reaches.

| stage | n | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|
| **total** | 56 | **7452** | **14128** | 17638 |
| model_generate | 43 | 5653 | 11661 | 13874 |
| output_safety | 43 | 1082 | 3014 | 4067 |
| input_safety | 56 | 894 | 2052 | 2904 |
| memory_chunk_embedding | 56 | 20.9 | 84.0 | 182 |
| memory_query_embedding | 43 | 18.5 | 96.9 | 286 |
| patch_apply | 43 | 6.7 | 11.2 | 24.5 |
| memory_search | 43 | 4.0 | 8.5 | 17.2 |
| save_memory_chunks | 56 | 3.7 | 7.1 | 20.9 |
| story_lookup | 56 | 2.7 | 3.3 | 10.3 |
| save_turn | 56 | 2.6 | 6.7 | 11.7 |
| prompt_build | 43 | 0.2 | 0.3 | 0.7 |
| editing_policy | 43 | 0.0 | 0.4 | 0.5 |
| intent_route | 56 | 0.0 | 0.0 | 0.1 |

Three model calls are 99.6% of the turn. Everything the project actually wrote
— routing, the editing policy, patch application, the repository, memory search
— totals under 40 ms at p50. **There is no Python to optimise here.** The Rust
port in phases 4–5 should be justified as ownership and packaging, not speed;
rewriting a 40 ms budget in a faster language cannot move a 7.5 s turn.

### By intent

| intent | n | p50 ms | p95 ms |
|---|---:|---:|---:|
| ask_definition | 3 | 8455 | 9344 |
| edit_story | 8 | 8143 | 14128 |
| continue_story | 32 | 7732 | 13504 |
| ask_grammar | 2 | 6681 | 17638 |
| parent_command | 1 | 6403 | 6403 |
| create_image | 3 | 925 | 1887 |
| undo_last_change | 2 | 903 | 935 |
| unknown | 5 | 884 | 2156 |

The early-return paths cost ~900 ms and nothing else. That whole 900 ms is the
input safety model call, which runs before routing — see finding 5.

### Drift across a session

| position | n | total p50 | generate p50 |
|---|---:|---:|---:|
| first third | 18 | 7642 | 5667 |
| middle third | 18 | 7161 | 5623 |
| last third | 20 | 7286 | 5562 |

**Flat.** Turn cost does not grow as the story accumulates over 56 turns. The
phase 5 story compiler is still worth building for prompt-cost control, but
this corpus gives it no latency justification — do not sell it as a speed fix
without a longer-session measurement that shows otherwise.

## Speech to text

`base.en`, cpu/int8, on the 56 corpus transcripts synthesised at 16 kHz.

| metric | p50 | p95 | max |
|---|---:|---:|---:|
| stt latency | 221 ms | 234 ms | 465 ms |
| audio duration | 2.34 s | 3.00 s | 3.55 s |
| real-time factor | 0.10 | 0.32 | 0.58 |

Model load is 274 ms, once per process. **STT was the last estimated number in
the architecture doc; it is now measured, and it is not a problem.** 221 ms is
small next to the 900 ms VAD tail that precedes it:

```
end of speech -> transcript  =  900 ms VAD tail  +  221 ms STT  =  1121 ms
```

Phase 1's plan to drop the endpoint tail to ~250 ms saves ~650 ms — nearly
three times what STT costs in total.

## Findings

### 1. `safety_model` defaults to None, which silently means the 12B story model

> **Fixed 2026-08-23, after it bit a real session.** `Settings.safety_model` now
> defaults to `gemma4:e2b-mlx` in code rather than only in `.env.example`, so a
> machine with no `.env` no longer runs safety on the 12B. The finding stood for
> months because nothing failed loudly — until the 12B blocked "she deflates the
> toys sometimes when she bites them" as sexual content, in a story about a dog.


`Settings.safety_model` defaults to `None`. `OllamaClient.generate_json`
resolves `model=None` to `default_model`. There is no `.env` in the repo, so
every safety review has been running on `gemma4:12b-mlx`, not the
`qwen3.5:2b-mlx` that `.env.example` and the architecture doc both describe.

That is 894 ms + 1082 ms = ~2.0 s per turn on safety, against the ~740 ms the
architecture doc budgets for two reviews.

### 2. A second app holding a model on the GPU cost ~3x throughput

`gemma4:12b-mlx` was measured at both ~59 tok/s and ~20 tok/s during this
session on identical work with identical inputs. Within any single burst it was
stable to within 1 tok/s, so it was never sample noise.

**Cause: an MTPLX endpoint was running with a model loaded.** Killing it removed
the effect, and it has not recurred. Nothing in Yarnotron was responsible.

The practical lesson is that this machine has no isolation between whatever else
holds the GPU and the numbers this bench prints. Before recording a baseline,
check that nothing else has a model resident:

```sh
ollama ps                       # should list only what you expect
ps aux | sort -nrk3 | head      # look for other inference apps
```

Co-residency of Yarnotron's *own* two models was the first hypothesis, and it
was wrong.
An A/B/A/B test alternating the safety model in and out, warming each phase
separately (`scripts/bench_models.py`):

| phase | resident | median tok/s |
|---|---|---:|
| A1 story alone | gemma4 | 59.4 |
| B1 story + safety | gemma4, qwen3.5:2b | 58.9 |
| A2 story alone | gemma4 | 56.9 |
| B2 story + safety | gemma4, qwen3.5:2b | 58.2 |

**Ratio 0.99x — co-residency is free.** The earlier measurement that suggested
a 3.3x penalty was confounded by ordering: it timed the "alone" case while the
story model was still warming, then the "co-resident" case once it was hot.
`bench_models.py` now alternates and warms each phase to prevent that.

So phase 1's plan to pin both models with `keep_alive` is **not** contradicted
— on this evidence it is safe, and the 640 ms swap it saves is real.

The spread did not correlate with thermal state (no warnings logged) or system
memory pressure (87% free), which is what made it look mysterious. It was
another process holding the GPU the whole time.

**Treat the stage proportions in this baseline as reliable and the absolute p50
as approximate.** Re-run the bench in the same session as any comparison rather
than diffing against numbers recorded on another day.

### 3. The documented safety model is unusable, for two separate reasons

With `.env.example` applied as written (`safety_model=qwen3.5:2b-mlx`), turns
fail with `OllamaUnavailableError`. The 2B model emits malformed JSON — a stray
quote after a null, which is not truncation and which no regex recovers:

```json
{ "action": "allow", ..., "parent_note": null" }
```

Scored properly over 68 real payloads (`scripts/bench_safety_models.py`), 7 of
68 safety calls are unrecoverable — about 10%. With two safety calls per turn
that is roughly one turn in five, not the "about half" an early four-turn
sample suggested.

The larger problem is judgement. The same run shows the 2B model **falsely
flagging 15 of 68 benign turns**. A child would be told "let's steer that in a
safer direction" on nearly a quarter of ordinary storytelling.

**The configuration shipped in `.env.example` does not work.** See the safety
model comparison below for what to use instead.

### 4. The JSON fence fallback is load-bearing on every draft call

`gemma4:12b-mlx` wraps its output in a ```` ```json ```` fence *even with
`format: json` set*. Every single story-draft call fails `json.loads` and is
recovered by the `_extract_json_object` regex in `services/ollama.py`. Phase 1
lists this as a detail to handle while adding streaming; it is better described
as the current parse path for 100% of drafts. An incremental streaming reader
must handle the fence from the first token or nothing will parse.

### 5. Input safety runs before routing, so "huh" costs 900 ms

`TurnProcessor.process()` calls `safety.review_transcript()` at line 51, before
`IntentRouter.route()` at line 53. Every turn pays a full safety model call,
including the ones that return a canned string without ever reaching the model
— `unknown` (884 ms p50), `undo_last_change` (903 ms), `create_image` (925 ms).

Routing first would make those paths effectively instant. Phase 1 already plans
to run input safety concurrently with generation; on early-return paths there
is no generation to overlap with, so ordering is the fix there, not concurrency.

### 6. Thinking is off in the turn path, but the plain `generate()` was a trap

Both turn-path call sites go through `generate_json`, which defaults
`think=False`, and Ollama honours it. `OllamaClient.generate()` defaulted to
`think=None`, which sends nothing and lets the model decide. From
`scripts/bench_models.py`:

| model | think | wall ms | eval tok | tok/s | thinking chars |
|---|---|---:|---:|---:|---:|
| gemma4:12b | `false` | 4122 | 219 | 60.0 | 0 |
| gemma4:12b | omitted | 9261 | 512 (capped) | 60.2 | 0 |
| gemma4:12b | `true` | 8924 | 512 (capped) | 57.9 | 1860 |
| qwen3.5:2b | `false` | 1412 | 142 | 186.7 | 0 |
| qwen3.5:2b | omitted | 2757 | 512 (capped) | 187.3 | 2042 |
| qwen3.5:2b | `true` | 2743 | 512 (capped) | 187.8 | 2042 |

Setting `think=False` explicitly roughly halves latency for both models. Note
the mechanism differs: qwen3.5 genuinely thinks when the flag is omitted (2042
chars), while gemma4 emits no thinking field but still runs to the 512-token
`num_predict` cap instead of stopping naturally at 219. Either way, omitting the
flag exhausts the token budget, which is a direct route to truncated JSON.

Throughput is unchanged in every case — this is a token-count effect, not a
speed effect. `generate()` now defaults to `think=False` to match
`generate_json`.

### 7. STT errors silently change intent

`IntentRouter` matches substrings on the raw transcript, so a transcription
slip reroutes the turn. 22 of 56 transcripts came back different from the input,
and at least two change behaviour:

| spoken | heard | intended | actual |
|---|---|---|---|
| "undo that" | "Do that." | `undo_last_change` | `continue_story` |
| "Add a scene where..." | "At a scene where..." | `edit_story` | `continue_story` |

A child asking to undo gets a story continuation. Invented proper nouns fare
badly too — Wren → "Ren", Fennick → "Fenak", Mackerel survives but Barnacle's
possessive does not. This is not in the architecture doc's risk list and it is
not fixed by any phase as written. A larger STT model, biasing decoding toward
the story's own character names, or confirming destructive intents would each
help.

## Choosing a safety model

Scored with `scripts/bench_safety_models.py` over 68 cases: 12 hand-written
probes (7 benign, 5 hazards that deliberately avoid the keyword guard's term
list) plus all 56 real output-review payloads replayed from the golden corpus.
"agrees" is agreement with the action the 12B reviewer recorded.

| model | usable JSON | agrees | **MISSED** | noisy | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| qwen3.5:0.8b-mlx | 68/68 (100%) | 59/68 | **9** | 0 | 149 | 198 |
| qwen3.5:2b-mlx | 61/68 (90%) | 45/68 | 1 | **15** | 382 | 577 |
| **gemma4:e2b-mlx** | 66/68 (97%) | **62/68** | 3 | 1 | **346** | 495 |
| gemma4:12b-mlx | 67/68 (99%) | 65/68 | 2 | 0 | 985 | 1807 |

`MISSED` = expected a flag, model allowed it. `noisy` = flagged something
benign. For a safety stage these are not symmetric: a miss ships unsafe content
to a child, a false flag is only annoying.

**Recommendation: `gemma4:e2b-mlx`.** Best judgement of any small model, within
three calls of the 12B, at 2.8x the speed. Swapping it in for the two safety
reviews saves roughly 1.3 s per turn against today's accidental 12B behaviour.

**Do not use `qwen3.5:0.8b-mlx`**, which phase 1 names for the per-sentence
output gate. It returns flawless JSON, fastest of everything measured, and
missed 9 hazards including all five synthetic ones — the grooming scenario, the
weapon instructions and the eating-disorder prompt were all allowed. It
essentially always answers `allow`. This is the worst failure mode available:
it looks perfect on every metric except the only one that matters.

### Two things that constrain any choice

**Ollama's MLX runner ignores `format: <json-schema>`.** Grammar-constrained
decoding would guarantee valid JSON from any model and make this whole question
moot. It does not work here: a schema demanding
`{"marker": "ZZZQQQ", "count": <int>}` returns prose about the weather instead,
on every MLX model tested. Only `format: "json"` — a hint, not a constraint —
has any effect. `bench_safety_models.py --schema` re-checks this after an
Ollama upgrade, or on a GGUF model where it should work.

**Every gemma model fences its JSON**, 65 of 68 calls for `e2b` and 68 of 68 for
`12b`, so `_extract_json_object` is what makes them usable at all. That is
already true of the story model today, so it adds no new risk — but it means a
streaming reader in phase 1 has to handle the fence from the first token.

### Worth trying before phase 1 commits

None of these is a purpose-built safety classifier; they are general models
asked to fill in a schema. `ShieldGemma` and `Llama-Guard` are trained for
exactly this and emit structured verdicts natively. Pull one and run it through
`bench_safety_models.py` before building the phase 1 safety pipeline around a
general model.

## What this means for the plan

- The **6.7 s** figure in the architecture doc is close but optimistic. Backend
  alone is **7.45 s** at p50, and end-to-end adds ~1.12 s of VAD tail and STT
  before the backend is even called.
- Phase 1's streaming work is aimed at the right target: three sequential model
  calls are the entire budget.
- Phase 1's per-sentence output gate on a 0.8B model must not be built as
  written: that model allows everything. `gemma4:e2b-mlx` is the working
  substitute, and it is 346 ms rather than 149 ms, which changes the budget for
  a per-sentence gate. Pinning both models is fine (finding 2).
- The 3x throughput spread (finding 2) was another app holding a model on the
  GPU, since removed. Check `ollama ps` before recording anything.
- Phases 4–5 cannot be justified on latency. Non-model work is under 40 ms.
