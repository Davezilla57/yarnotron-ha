# Yarnotron steward: a Hermes profile

A Hermes agent profile that *operates* Yarnotron for the grown-ups: status, launch, shutdown and safety
alerts. It cannot talk to the child; the child's voice path stays Home Assistant → Yarnotron backend.

```
 parent ⇄ Hermes (profile "yarnotron") ──MCP stdio──▶ yarnotron_mcp.py ──▶ Home Assistant services ─▶ host supervisor ─▶ Ollama · Whisper · backend
                                                             └──────────────▶ Yarnotron backend (/v1/status, /v1/safety-events)
 child  ⇄ voice satellite ⇄ Home Assistant Assist ⇄ Yarnotron conversation agent ⇄ backend      (Hermes is not in this path)
```

## Contents

| Path | What |
|---|---|
| `yarnotron_mcp.py` | Stdio MCP server, Python 3.11+, standard library only. Copy this one file to the Hermes machine |
| `profile/SOUL.md` | The profile's persona and rules |
| `profile/config.snippet.yaml` | The `mcp_servers` block to merge into the profile's `config.yaml` |
| `profile/skills/yarnotron-ops/` | Skill: which tool for which request, and an error-code table |
| `tests/` | 15 tests against stdlib fake servers and the real stdio protocol |

## Install

Steps marked **[Hermes machine]** happen where Hermes runs; **[HA]** is the Home Assistant web page.

1. **[HA]** Profile (bottom-left) → Security → Long-lived access tokens → Create token. Copy it.
   Use a Home Assistant user that is allowed to call `yarnotron.*` services.
2. **[Hermes machine]** Copy `yarnotron_mcp.py` somewhere stable, for example `~/jeevus/mcp/yarnotron_mcp.py`.
3. **[Hermes machine]** Create the profile: `hermes profile create yarnotron` (or use the dashboard's Profile Builder).
4. **[Hermes machine]** Copy `profile/SOUL.md` over the new profile's `SOUL.md`.
5. **[Hermes machine]** Merge `profile/config.snippet.yaml` into the profile's `config.yaml`. Fill in the
   path, the backend URL and token, and `HASS_URL` / `HASS_TOKEN`.
6. **[Hermes machine]** Put `profile/skills/yarnotron-ops/` in the profile's skills folder (default location
   `~/.hermes/skills/<category>/yarnotron-ops/SKILL.md`), or add its parent to `skills.external_dirs`.
7. Start a new session in the profile and ask: *"What's Yarnotron's status?"*

I could not verify which folder a named profile reads skills from; the Hermes documentation I found
describes `~/.hermes/skills/` and `skills.external_dirs`. If the profile does not see the skill, use
`external_dirs`.

## Tools

| Tool | Reads/changes | Guard |
|---|---|---|
| `status` | Reads | Returns problems with explanation, observed detail and what to try |
| `launch` | Starts anything not answering | Safe to repeat. Wakes a sleeping host computer when Home Assistant has its MAC address |
| `shutdown` | Saves the story, stops the host side | `confirmed=true` required. `force` also needs `YARNOTRON_MCP_ALLOW_FORCE=1` |
| `safety_events` | Reads | No transcript, no parent note |
| `review_event` | Marks reviewed, recorded as `YARNOTRON_MCP_ACTOR` | `confirmed=true` |
| `release_event` | Releases held-back content | Off unless `YARNOTRON_MCP_ALLOW_RELEASE=1`; `confirmed=true` |
| `list_stories` | Reads | Titles (80 chars) and counts, no text |
| `story_text` | Reads a story as Markdown | Off unless `YARNOTRON_MCP_ALLOW_STORY_TEXT=1`; `confirmed=true` |
| `digest` | Reads | Counts only; suited to a scheduled run |

`confirmed` is a convention the model follows, not something the server can check. The real controls
are the `tools.include` list, the ALLOW flags, and the Home Assistant user's permissions. For
anything you would not want done on a model's misjudgement, leave the tool out of `include`.

## Why Home Assistant is the preferred route

Home Assistant keeps its own record of the child's session and a "powered down" flag. When
`HASS_URL` is set, launch/shutdown/status go through the `yarnotron.*` services, so the voice agent
says the right thing afterwards. Without it the server calls the host supervisor directly; that
works, but Home Assistant is not told (the result says so), and Wake-on-LAN is unavailable.

## Weekly digest

A scheduled Hermes task in this profile, with a prompt like: *"Run the weekly Yarnotron digest: call the
digest and status tools and write four lines: counts by action, categories seen, how many are
unreviewed, anything unhealthy. No quotes or story content."* Check your Hermes version's scheduling
syntax; I did not verify it.

## Safety notes

- The child's words never appear in any tool result. Keep it that way if you extend the server.
- Hermes profiles do not sandbox the filesystem. Do not run this profile on a machine or account that
  can read the Yarnotron database.
- Story titles and error text are untrusted input to the model. The persona says to treat them as data.
- Hermes does not pass your shell environment to MCP servers; every variable must be in `config.yaml`,
  which therefore holds two tokens. Protect that file.

## Tests

`pytest integrations/hermes/tests`

Not verified: a live Hermes install, the exact response format of Home Assistant's `?return_response`
on your version (tested against a fake that follows the documented shape), and long-running tool calls
under Hermes's `timeout` setting. Set the server's `timeout` above the longest launch (a few minutes).
