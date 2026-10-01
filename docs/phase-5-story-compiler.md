# Phase 5: the story compiler

Phase 5 is mostly the Rust port, which is deferred for the same reason as phase
4: non-model work is under 40 ms, so the language buys no latency. One item in
it was not about Rust at all —

> add the story compiler the architecture doc always wanted: a bounded, compact
> context view instead of the whole document, so prompt cost stops growing with
> story length

— and by phase 1 that had become the largest single component of time to first
audio. It was pulled forward and done.

## Why it was worth pulling forward

The prompt inlined the entire story document, pretty-printed. Measured:

| scenes | prompt chars | of which story | static share |
|---:|---:|---:|---:|
| 2 | 7,531 | 2,008 | 57% |
| 10 | 11,828 | 6,305 | 37% |
| 40 | 30,041 | 24,518 | **14%** |

The static instructions are 4,325 characters and constant, and Ollama caches
that prefix. Everything that grows is the story, so a child who has been
writing for an hour pays for it on every single turn. That is backwards: the
reward for writing more should not be waiting longer.

## What it keeps, and why

| | treatment | reason |
|---|---|---|
| every character | always, notes clipped | patches address them by id, and `StoryEditingPolicy` dedupes against this list — dropping one causes duplicate characters and unresolvable `update_character` |
| recent 6 scenes | in full, with draft text | that is where the child is writing |
| older scenes | id, title and sequence only | still targetable by `update_scene`, without carrying draft text |
| recent 15 facts | in full | |
| anything elided | **declared**, e.g. `earlier_scenes_not_shown: 12` | a model that cannot see a scene and is not told it exists will add it again |

`indent=2` was also dropped. Eighteen percent of the story block was whitespace
the model had to read and gained nothing from.

## What it saves

Same session, alternating prompts so machine drift cannot masquerade as a
result, on the actual story the corpus builds (16 scenes, 3 characters, 7
facts):

| | before | after |
|---|---:|---:|
| prompt tokens | 2,991 | 2,014 |
| prompt_eval | 1084 ms | **521 ms** |

**52% off prompt evaluation, about 560 ms per turn**, on a story of the size a
child reaches in one sitting.

It compounds with story length, which is the actual point:

| scenes | before | after | saved |
|---:|---:|---:|---:|
| 5 | 1398 ms | 1186 ms | 15% |
| 20 | 2849 ms | 1559 ms | 45% |
| 40 | 6234 ms | 2250 ms | 64% |
| 80 | **13343 ms** | 2964 ms | **78%** |

At eighty scenes the old prompt cost **thirteen seconds** of evaluation before a
single token was generated. A long-running story was quietly becoming unusable,
and the phase 0 drift table missed it because the corpus only reaches about
twenty scenes.

## Did it change what the model decides?

Full corpus, 56/56 completed:

| field | identical |
|---|---|
| intent | 56/56 |
| safety action | 56/56 |
| patch shape | 46/56 |

The ten patch-shape differences go in both directions, and three of them are
the model now choosing `update_scene` where it previously chose `add_scene` —
which is what the prompt's own deduplication instruction asks for. Given the
corpus cascades (one divergence changes every later prompt, see
`docs/phase-1-streaming.md`), this is within normal drift.

The specific risk the compiler introduces is different and worth testing
directly: **does hiding scenes make the model re-add them?** Comparing the final
story from each run:

| | scenes | characters | facts | duplicate titles |
|---|---:|---:|---:|---|
| whole document | 18 | 3 | 6 | none |
| compiled view | 16 | 3 | 7 | none |

No duplicates, and with a six-scene window against sixteen scenes, elision was
genuinely exercised. The declared `earlier_scenes` list is doing its job.

## Caveat on the end-to-end number

The corpus run's `first_audio` and `prompt_eval` should not be compared against
earlier runs. Cross-session comparison on this machine has been unreliable all
the way through, and the controlled same-session measurement above is the one
to trust. What can be said end to end is that the corpus completed 56/56 with
unchanged routing and safety decisions.

## What phase 5 still owes

The Rust port. It is worth doing for ownership and packaging, not speed, and it
now inherits a bounded prompt rather than one that grows without limit — which
is a better thing to port.
