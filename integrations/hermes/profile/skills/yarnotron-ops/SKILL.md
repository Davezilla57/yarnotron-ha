---
name: yarnotron-ops
description: Operate and troubleshoot Yarnotron (the child's voice storytelling system): check status, launch or shut down the Mac side, and work through safety alerts. Use for any question about Yarnotron running, stopping, failing, or flagging something.
version: 0.1.0
platforms: [macos, linux]
---

# Operating Yarnotron

Tools arrive as `mcp_yarnotron_<name>`, for example `mcp_yarnotron_status`. Never ask for or reconstruct what the child said.

## Which tool

| The parent says | Do this |
|---|---|
| "Is it working?" / anything seems off | `status`, then explain each problem using its `explanation`, `observed` and `try` |
| "Start it" / "it's not responding" | `status` first, then `launch`. Launch only starts what is not answering, so it is safe to repeat |
| "Turn it off" / "put it away" | Say what will happen (the current story is saved, running work finishes, Ollama, speech-to-text and the backend are stopped, and a child mid-story is interrupted). Ask for a yes. Only then `shutdown` with `confirmed=true` |
| "Anything I should know?" | `digest`, then `safety_events` with `unreviewed_only=true` |
| "I've looked at that alert" | Confirm which one, then `review_event` with `confirmed=true` |

## Reading a status result

- `problems[]` entries have `where`, `explanation`, sometimes `observed` (raw detail) and `try`.
  Report the explanation first, then the suggestions in the order given.
- `backend.shutting_down: true` means a shutdown started and did not finish. The backend is refusing
  new stories. `launch` clears it.
- `backend.latest_session_strikes` against `strike_limit` shows how close a session is to ending.
  A session that has ended stays locked until a parent starts a new one in Home Assistant (button
  "Start new session" or the `yarnotron.start_new_session` service). You cannot do that from here.

## Alerts

- `action`: `redirect` (steered the child gently, may count as a strike), `block` or `alert_parent`
  (the turn was stopped). `severity` ranges from `info` to `urgent`.
- `counted_as_strike` events add up to the strike limit, then the session ends.
- Describe by category and action only. Offer to mark reviewed. Offer the dashboard for the words.
- Releasing held-back content is a parent decision made in the dashboard unless this profile has
  `release_event` enabled; even then it needs an explicit yes.

## Failure vocabulary

See `references/error-codes.md` for the Mac supervisor's codes and what usually fixes each one.

## Weekly digest (for a scheduled run)

Call `digest` and `status`. Write four lines at most: counts by action, categories
seen, unreviewed count, anything unhealthy. No names, no quotes, no story content.
