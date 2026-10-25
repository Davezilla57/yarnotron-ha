# Agent engine (`YARNOTRON_ENGINE=agent`)

Default is `classic`, the turn pipeline. `agent` replaces the intent router, the one-shot JSON draft and the patch
plumbing with a **Talker** (streams plain speech from `agent/spec/constitution.md`, no tools) and a **Scribe**
(proposes story actions as JSON, validated against `agent/spec/tools.schema.json`). Everything deterministic stays
in code and runs before or after the agent: keyword guard, strikes and session lock, parent release, the
yes/no resolver for held actions, per-sentence gate, full output review before anything is written, deferred
input review, and removal of addresses, email addresses and phone numbers (`services/pii.py`, both engines).

Same HTTP contract as classic, so the Home Assistant integration needs no change.

| Setting | Default | Meaning |
|---|---|---|
| `YARNOTRON_ENGINE` | `classic` | `classic` or `agent` |
| `YARNOTRON_AGENT_SCRIBE_BUDGET_SECONDS` | `6` | Past this the Scribe's answer is dropped and the reply stands |

Rules enforced in code, not prompt: a removal, a replacement of the child's wording or an unrequested edit is held
for the child's next-turn yes; a scene from ordinary narration is offered ("Want me to save that as its own scene?")
and added only on yes (not repeated within 4 turns); saved wording must be a quote of the child's own words; picture
requests only create a pending parent-approval asset; undo drops every other action; at most 3 adds, 1 interest and
8 actions per turn; unknown tools are refused. Worried-child lines (self-harm keywords, model categories
`self_harm`/`worried_child`) are the fixed script in `agent/code_messages.py`, which is a draft awaiting parent approval.

Not yet done (see jeevus `docs/yarnotron-agent/`): the evaluation harness, a fast first-sentence classifier, model-flagged
worried-child categories in the input review, and tuning against the classic engine.
