# Bench

The phase 0 baseline harness. Every later phase is judged against the numbers
these scripts print, so treat the corpus as frozen: changing `corpus.json`
invalidates every recorded baseline and every golden response.

## Prerequisites

```sh
ollama serve        # gemma4:12b-mlx, gemma4:e2b-mlx, nomic-embed-text:v1.5
cp .env.example .env
```

The datastore is a SQLite file created on first use; there is nothing to start.

## Turn latency and golden output

```sh
uv run python scripts/bench_turns.py                 # one pass, stage table
uv run python scripts/bench_turns.py --repeat 3      # percentiles worth trusting
uv run python scripts/bench_turns.py --stream --seed 42 --write-golden
uv run python scripts/bench_turns.py --limit 5       # quick smoke run
```

The script drives `TurnProcessor.process()` in-process rather than over HTTP,
which is what makes the per-stage numbers available: it captures the
`timing {...}` line `LatencyTimer.log()` already emits and aggregates it. HTTP
framing on `POST /v1/agent/turn` is a sub-millisecond loopback cost and is not
included.

Each pass seeds a fresh copy of the fixed story, replays all 56 turns in order,
then deletes it, so passes are independent and repeatable.

## Golden responses

`golden/responses.json` is the reference output to diff against after every
future change, and eventually against the Rust port. Generated ids and
timestamps are replaced with stable per-response ordinals (`<character_2>`), so
a patch that targets the second character still reads that way and the file
only changes when behaviour changes.

Regenerate deliberately, never as a matter of course:

```sh
uv run python scripts/bench_turns.py --stream --seed 42 --write-golden
uv run python scripts/diff_golden.py before.json after.json
```

**Always pass `--stream --seed` together.** The seed has no effect otherwise:
Ollama's MLX runner ignores `seed` when `stream` is false. Measured over eight
prompts, three runs each, same seed — buffered reproduced itself 1/8, streamed
8/8.

### What this corpus can and cannot tell you

It **cannot** be used as a byte-exact oracle, even streamed and seeded. Two
identical streamed runs still diverge: the first turn's reply matches for 138
characters and then forks, which is floating-point nondeterminism flipping a
token late in a long generation. Because the corpus threads one story through
all 56 turns, that single fork changes every prompt after it, and the
divergence compounds — 42/56 turns differ in text, 16/56 in decisions, mostly
from one early fork rather than 16 real faults.

It **can** tell you that routing is unaffected (`intent` has been 56/56 in
every comparison run so far), and it is a good source of realistic payloads for
scoring a single component, which is how `bench_safety_models.py` uses it.

For a real A/B — phase 5's Rust port against Python — replay each turn against
fixed recorded story state instead of letting the corpus evolve, so turns are
independent and a divergence cannot cascade.

`diff_golden.py` separates decisions from prose. `intent`, `safety` and
`patch shape` are decisions; `response_text` and patch payloads carry generated
wording and will differ without meaning anything.

## Choosing a safety model

```sh
uv run python scripts/bench_safety_models.py                      # all candidates
uv run python scripts/bench_safety_models.py --cases synthetic    # hazard probes only
uv run python scripts/bench_safety_models.py --models X --repeat 5
uv run python scripts/bench_safety_models.py --schema             # recheck schema support
```

Scores candidates on 12 hand-written probes plus all 56 real output-review
payloads replayed from the golden corpus. The column that decides it is
`MISSED` — expected a flag, model allowed it. A model can score 100% on JSON
validity and still be unusable: `qwen3.5:0.8b-mlx` returns perfect JSON in
149 ms and allows every hazard in the set.

`--schema` tests `format: <json-schema>`, which would guarantee valid JSON if it
worked. Ollama's MLX runner ignores it, so it is off by default; re-check after
an Ollama upgrade or on a GGUF model.

## Speech-to-text

```sh
uv sync --extra voice --extra dev
uv run python scripts/bench_stt.py
```

Synthesises each corpus transcript with the macOS speech synthesiser at the
same 16 kHz the recorder uses, then transcribes it with the same
`FasterWhisperTranscriber` the voice loop uses. Synthesised speech is cleaner
than a child at a microphone, so the result is a floor rather than a typical
case.

## Corpus shape

56 turns over a fixed two-character, two-scene story:

| intent | turns |
|---|---|
| `continue_story` | 32 |
| `edit_story` | 8 |
| `unknown` | 5 |
| `ask_definition` | 3 |
| `create_image` | 3 |
| `ask_grammar` | 2 |
| `undo_last_change` | 2 |
| `parent_command` | 1 |

Three of the `continue_story` and `edit_story` turns (t53–t55) are written to
trip the keyword guard, so the redirect path is measured too.
