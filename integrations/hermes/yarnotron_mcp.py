#!/usr/bin/env python3
"""Yarnotron MCP server for a Hermes parent/ops profile.

Speaks the Model Context Protocol over stdio (newline-delimited JSON-RPC) using
only the standard library, so it runs anywhere Python 3.11+ does -- copy this one
file to the machine that runs Hermes.

It lets an agent *operate* Yarnotron: check status, launch and shut down the
Mac side, and handle safety alerts. It deliberately cannot talk to the child:
there is no tool that sends a story turn, and a child's words never appear in
any result (alerts are stripped of transcripts and parent notes).

Environment
  YARNOTRON_URL                  backend, e.g. http://192.168.1.20:8989      (required)
  YARNOTRON_API_TOKEN            backend bearer token, if the backend uses one
  YARNOTRON_CHILD_ID             child to report on (default local_child)
  YARNOTRON_SUPERVISOR_URL       Mac supervisor, e.g. http://192.168.1.20:8990
  YARNOTRON_SUPERVISOR_TOKEN     its bearer token
  HASS_URL, HASS_TOKEN           Home Assistant; when set, status/launch/shutdown go
                                 through the Yarnotron integration so Home Assistant's
                                 own session state stays correct (preferred)
  YARNOTRON_MCP_ACTOR            name recorded on reviews (default hermes)
  YARNOTRON_MCP_ALLOW_FORCE      1 to permit forced shutdown            (default 0)
  YARNOTRON_MCP_ALLOW_RELEASE    1 to permit releasing held content     (default 0)
  YARNOTRON_MCP_ALLOW_STORY_TEXT 1 to permit reading story text         (default 0)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SERVER_NAME = "yarnotron"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
HTTP_TIMEOUT = 15


class ToolError(Exception):
    """A failure to report to the model, in words that explain it."""


@dataclass
class Config:
    backend_url: str
    backend_token: str | None
    child_id: str
    supervisor_url: str | None
    supervisor_token: str | None
    hass_url: str | None
    hass_token: str | None
    actor: str
    allow_force: bool
    allow_release: bool
    allow_story_text: bool

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        e = os.environ if env is None else env

        def flag(name: str) -> bool:
            return e.get(name, "0").strip().lower() in {"1", "true", "yes"}

        def url(name: str) -> str | None:
            value = (e.get(name) or "").strip().rstrip("/")
            if value and "://" not in value:
                value = f"http://{value}"
            return value or None

        backend = url("YARNOTRON_URL")
        if not backend:
            raise ToolError("YARNOTRON_URL is not set.")
        return cls(
            backend_url=backend,
            backend_token=(e.get("YARNOTRON_API_TOKEN") or "").strip() or None,
            child_id=e.get("YARNOTRON_CHILD_ID", "local_child"),
            supervisor_url=url("YARNOTRON_SUPERVISOR_URL"),
            supervisor_token=(e.get("YARNOTRON_SUPERVISOR_TOKEN") or "").strip() or None,
            hass_url=url("HASS_URL"),
            hass_token=(e.get("HASS_TOKEN") or "").strip() or None,
            actor=e.get("YARNOTRON_MCP_ACTOR", "hermes"),
            allow_force=flag("YARNOTRON_MCP_ALLOW_FORCE"),
            allow_release=flag("YARNOTRON_MCP_ALLOW_RELEASE"),
            allow_story_text=flag("YARNOTRON_MCP_ALLOW_STORY_TEXT"),
        )

    @property
    def via_home_assistant(self) -> bool:
        return bool(self.hass_url and self.hass_token)


# ------------------------------------------------------------------- HTTP


def request(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: Any = None,
    params: dict[str, Any] | None = None,
    timeout: float = HTTP_TIMEOUT,
    what: str = "service",
) -> Any:
    """JSON request. Failures become ToolError sentences that say what is going on."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:
        text = err.read().decode(errors="replace")[:300]
        if err.code in (401, 403):
            raise ToolError(f"The {what} rejected the token (HTTP {err.code}).") from err
        raise ToolError(f"The {what} answered HTTP {err.code}: {text}") from err
    except urllib.error.URLError as err:
        reason = getattr(err, "reason", err)
        if isinstance(reason, ConnectionRefusedError):
            raise ToolError(
                f"Connection refused by the {what} at {url.split('?')[0]}: "
                "the machine is up but nothing is listening (the service is not running)."
            ) from err
        if isinstance(reason, TimeoutError):
            raise ToolError(
                f"No answer from the {what} at {url.split('?')[0]} within {timeout:g}s: "
                "the machine may be asleep, off, or off the network."
            ) from err
        raise ToolError(f"Could not reach the {what} at {url.split('?')[0]}: {reason}") from err
    except TimeoutError as err:
        raise ToolError(f"The {what} did not answer within {timeout:g}s.") from err
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError as err:
        raise ToolError(f"The {what} did not return JSON.") from err


class Clients:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def backend(self, method: str, path: str, **kw: Any) -> Any:
        return request(
            method,
            self.cfg.backend_url + path,
            token=self.cfg.backend_token,
            what="Yarnotron backend",
            **kw,
        )

    def backend_health(self) -> Any:
        return request("GET", self.cfg.backend_url + "/health", what="Yarnotron backend")

    def supervisor(self, method: str, path: str, **kw: Any) -> Any:
        if not self.cfg.supervisor_url:
            raise ToolError(
                "No Mac supervisor is configured (set YARNOTRON_SUPERVISOR_URL), "
                "and Home Assistant is not configured either."
            )
        return request(
            method,
            self.cfg.supervisor_url + path,
            token=self.cfg.supervisor_token,
            what="Mac supervisor",
            **kw,
        )

    def home_assistant(self, service: str, data: dict[str, Any], timeout: float) -> dict[str, Any]:
        url = f"{self.cfg.hass_url}/api/services/yarnotron/{service}?return_response"
        reply = request(
            "POST",
            url,
            token=self.cfg.hass_token,
            body=data,
            timeout=timeout,
            what="Home Assistant",
        )
        response = reply.get("service_response") if isinstance(reply, dict) else None
        if not isinstance(response, dict):
            raise ToolError(
                f"Home Assistant accepted yarnotron.{service} but returned no response. "
                "Is the Yarnotron integration loaded?"
            )
        return response


# ------------------------------------------------------------- redaction

_EVENT_KEEP = ("id", "created_at", "child_id", "story_id", "session_id", "counted_as_strike")


def redact_event(event: dict[str, Any]) -> dict[str, Any]:
    """An alert without what the child said and without the parent-facing note.

    Both are the child's words in one form or another, and anything returned here
    lands in an agent's context and memory.
    """
    safety = event.get("safety") or {}
    out = {k: event.get(k) for k in _EVENT_KEEP}
    out.update(
        action=safety.get("action"),
        severity=safety.get("severity"),
        categories=safety.get("categories") or [],
        reviewed=event.get("reviewed_at") is not None,
    )
    return out


# ------------------------------------------------------------------ tools


def _require_confirmed(args: dict[str, Any], what: str) -> None:
    if args.get("confirmed") is not True:
        raise ToolError(
            f"Not done: {what} needs the user's explicit yes. Ask them in plain words, "
            "and only after they agree call this again with confirmed=true."
        )


def tool_status(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    cfg = c.cfg
    result: dict[str, Any] = {"problems": []}

    def attempt(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except ToolError as err:
            result["problems"].append({"where": name, "explanation": str(err)})
            return None

    if cfg.via_home_assistant:
        report = attempt("Home Assistant", lambda: c.home_assistant("status", {}, 60))
        if report:
            result["components"] = [
                {k: comp.get(k) for k in ("name", "side", "state", "detail")}
                for comp in report.get("components", [])
            ]
            result["spoken_summary"] = report.get("summary")
            for comp in report.get("components", []):
                issue = comp.get("issue")
                if issue:
                    result["problems"].append(
                        {
                            "where": comp["name"],
                            "explanation": issue["summary"],
                            "observed": issue.get("detail"),
                            "try": issue.get("hints", []),
                        }
                    )
    elif cfg.supervisor_url:
        sup = attempt("Mac supervisor", lambda: c.supervisor("GET", "/v1/status"))
        if sup:
            result["components"] = [
                {"name": s["label"], "side": "mac", "state": s["state"], "detail": s["detail"]}
                for s in sup.get("services", [])
            ]
            for s in sup.get("services", []):
                if not s["ok"]:
                    result["problems"].append(
                        {"where": s["label"], "explanation": s["detail"], "log": s.get("log")}
                    )

    health = attempt("Yarnotron backend", c.backend_health)
    result["backend_answering"] = health is not None
    if health is not None:
        status = attempt(
            "Yarnotron backend status",
            lambda: c.backend("GET", "/v1/status", params={"child_id": cfg.child_id}),
        )
        if status:
            child = status.get("child") or {}
            result["backend"] = {
                "version": status.get("version"),
                "shutting_down": status.get("shutting_down", False),
                "stories": status.get("stories"),
                "models": status.get("models"),
                "open_sessions": child.get("open_sessions"),
                "unreviewed_safety_events": child.get("unreviewed_safety_events"),
                "pending_media": child.get("pending_media"),
                "latest_session_strikes": child.get("latest_session_strikes"),
                "strike_limit": status.get("strike_limit"),
            }
    result["ok"] = not result["problems"] and result["backend_answering"]
    return result


def _job_result(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(job.get("ok")),
        "summary": job.get("summary"),
        "story_saved": job.get("story"),
        "steps": [
            {
                "service": s["label"],
                "outcome": s["outcome"],
                "message": s["message"],
                "warnings": s.get("warnings", []),
                "issue": s.get("issue"),
            }
            for s in job.get("steps", [])
        ],
    }


def _direct_job(c: Clients, path: str, params: dict[str, Any] | None, timeout: float) -> dict:
    job = c.supervisor("POST", path, params=params)
    deadline = time.monotonic() + timeout
    while job.get("state") != "done":
        if time.monotonic() > deadline:
            return {
                "ok": False,
                "summary": f"Still running after {timeout:g}s; check status later.",
                "steps": job.get("steps", []),
            }
        time.sleep(1.5)
        job = c.supervisor("GET", f"/v1/jobs/{job['id']}")
    return job


def tool_launch(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    wait = float(args.get("wait_seconds", 300))
    if c.cfg.via_home_assistant:
        response = c.home_assistant("launch", {"wait": True}, wait + 30)
        return {
            "via": "home_assistant",
            "ok": response.get("ok"),
            "summary": response.get("summary"),
            "details_markdown": response.get("details"),
        }
    job = _direct_job(c, "/v1/launch", None, wait)
    if job.get("ok"):
        status = c.backend("GET", "/v1/status", params={"child_id": c.cfg.child_id})
        if status.get("shutting_down"):
            c.backend("POST", "/v1/system/cancel-shutdown", body={})
    return {"via": "supervisor", **_job_result(job)}


def tool_shutdown(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    _require_confirmed(args, "shutting down Yarnotron (it saves the story and stops the Mac side)")
    force = bool(args.get("force", False))
    if force and not c.cfg.allow_force:
        raise ToolError(
            "Forced shutdown is disabled for this profile (YARNOTRON_MCP_ALLOW_FORCE=0). "
            "A forced stop can lose unfinished safety reviews. Ask the user to run it themselves."
        )
    wait = float(args.get("wait_seconds", 240))
    if c.cfg.via_home_assistant:
        response = c.home_assistant("shutdown", {"force": force, "wait": True}, wait + 30)
        return {
            "via": "home_assistant",
            "ok": response.get("ok"),
            "summary": response.get("summary"),
            "details_markdown": response.get("details"),
        }
    job = _direct_job(c, "/v1/shutdown", {"force": "1"} if force else None, wait)
    return {
        "via": "supervisor",
        "note": "Home Assistant was not involved, so its voice agent may still think a session is open.",
        **_job_result(job),
    }


def tool_safety_events(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    limit = max(1, min(int(args.get("limit", 10)), 100))
    events = c.backend("GET", "/v1/safety-events", params={"limit": 100})
    rows = [redact_event(e) for e in events if e.get("child_id", c.cfg.child_id) == c.cfg.child_id]
    if args.get("unreviewed_only"):
        rows = [r for r in rows if not r["reviewed"]]
    return {"events": rows[:limit], "note": "Transcripts and parent notes are withheld by design."}


def tool_review_event(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    _require_confirmed(args, "marking this safety event reviewed")
    event_id = str(args["event_id"])
    note = args.get("note")
    c.backend(
        "POST",
        f"/v1/safety-events/{urllib.parse.quote(event_id)}/review",
        body={"reviewed_by": c.cfg.actor, "review_note": note},
    )
    return {"reviewed": event_id, "reviewed_by": c.cfg.actor}


def tool_release_event(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    if not c.cfg.allow_release:
        raise ToolError(
            "Releasing held content is disabled for this profile (YARNOTRON_MCP_ALLOW_RELEASE=0). "
            "The parent should do this in the Yarnotron dashboard or Home Assistant."
        )
    _require_confirmed(args, "releasing content that Yarnotron held back from the child")
    event_id = str(args["event_id"])
    c.backend(
        "POST",
        f"/v1/safety-events/{urllib.parse.quote(event_id)}/release",
        body={"released_by": c.cfg.actor, "review_note": args.get("note")},
    )
    return {"released": event_id, "released_by": c.cfg.actor}


def tool_list_stories(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    stories = c.backend("GET", "/v1/stories")
    return {
        "stories": [
            {
                "id": s.get("id"),
                "title": str(s.get("title", ""))[:80],
                "updated_at": s.get("updated_at"),
                "scenes": len(s.get("scenes") or []),
                "characters": len(s.get("characters") or []),
            }
            for s in stories
        ]
    }


def tool_story_text(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    if not c.cfg.allow_story_text:
        raise ToolError(
            "Reading story text is disabled for this profile (YARNOTRON_MCP_ALLOW_STORY_TEXT=0). "
            "Story text is the child's writing; enable it only if you accept that it enters this "
            "agent's context and memory."
        )
    _require_confirmed(args, "reading the child's story text")
    story_id = urllib.parse.quote(str(args["story_id"]))
    req = urllib.request.Request(
        f"{c.cfg.backend_url}/v1/stories/{story_id}/export.md",
        headers={"Authorization": f"Bearer {c.cfg.backend_token}"} if c.cfg.backend_token else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as response:
            return {"markdown": response.read().decode()}
    except (urllib.error.URLError, TimeoutError) as err:
        raise ToolError(f"Could not export the story: {err}") from err


def tool_digest(c: Clients, args: dict[str, Any]) -> dict[str, Any]:
    """Counts only, so it can run unattended on a schedule."""
    rows = [redact_event(e) for e in c.backend("GET", "/v1/safety-events", params={"limit": 100})]
    rows = [r for r in rows if r["child_id"] == c.cfg.child_id]
    return {
        "events_in_window": len(rows),
        "by_action": dict(Counter(r["action"] for r in rows)),
        "by_category": dict(Counter(cat for r in rows for cat in r["categories"])),
        "strikes": sum(1 for r in rows if r["counted_as_strike"]),
        "unreviewed": sum(1 for r in rows if not r["reviewed"]),
        "stories": len(c.backend("GET", "/v1/stories")),
        "note": "The window is the most recent 100 events the backend returns.",
    }


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


CONFIRM = {
    "type": "boolean",
    "description": "Set true ONLY after the user explicitly said yes in this conversation.",
}

TOOLS: dict[str, dict[str, Any]] = {
    "status": {
        "description": "Status of every Yarnotron component on Home Assistant and on the Mac, "
        "plus backend counts. Problems come with an explanation, the observed detail and what to try.",
        "inputSchema": _schema(),
        "fn": tool_status,
        "readOnly": True,
    },
    "launch": {
        "description": "Check the Mac side and start anything not answering (wakes a sleeping Mac "
        "when Home Assistant is configured with its MAC address). Safe to repeat.",
        "inputSchema": _schema({"wait_seconds": {"type": "number", "default": 300}}),
        "fn": tool_launch,
        "readOnly": False,
    },
    "shutdown": {
        "description": "Save the current story, drain running work, then stop the Mac side. "
        "Interrupts any child currently playing. Needs the user's explicit yes.",
        "inputSchema": _schema(
            {
                "confirmed": CONFIRM,
                "force": {"type": "boolean", "default": False},
                "wait_seconds": {"type": "number", "default": 240},
            },
            ["confirmed"],
        ),
        "fn": tool_shutdown,
        "readOnly": False,
        "destructive": True,
    },
    "safety_events": {
        "description": "Recent safety events (action, severity, categories, time, reviewed). "
        "Never includes what the child said.",
        "inputSchema": _schema(
            {"limit": {"type": "integer", "default": 10}, "unreviewed_only": {"type": "boolean"}}
        ),
        "fn": tool_safety_events,
        "readOnly": True,
    },
    "review_event": {
        "description": "Mark one safety event as reviewed by a parent.",
        "inputSchema": _schema(
            {"event_id": {"type": "string"}, "note": {"type": "string"}, "confirmed": CONFIRM},
            ["event_id", "confirmed"],
        ),
        "fn": tool_review_event,
        "readOnly": False,
    },
    "release_event": {
        "description": "Release content Yarnotron held back from the child. Disabled unless the "
        "profile enables it; needs the user's explicit yes.",
        "inputSchema": _schema(
            {"event_id": {"type": "string"}, "note": {"type": "string"}, "confirmed": CONFIRM},
            ["event_id", "confirmed"],
        ),
        "fn": tool_release_event,
        "readOnly": False,
        "destructive": True,
    },
    "list_stories": {
        "description": "Stories with id, title, last update and counts. No story text.",
        "inputSchema": _schema(),
        "fn": tool_list_stories,
        "readOnly": True,
    },
    "story_text": {
        "description": "Export one story as Markdown. Disabled unless the profile enables it.",
        "inputSchema": _schema(
            {"story_id": {"type": "string"}, "confirmed": CONFIRM}, ["story_id", "confirmed"]
        ),
        "fn": tool_story_text,
        "readOnly": True,
    },
    "digest": {
        "description": "Counts of recent safety events by action and category, strikes, "
        "unreviewed and stories. Contains no text; suited to a weekly scheduled summary.",
        "inputSchema": _schema(),
        "fn": tool_digest,
        "readOnly": True,
    },
}


# --------------------------------------------------------------- protocol


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
            "annotations": {
                "readOnlyHint": spec.get("readOnly", False),
                "destructiveHint": spec.get("destructive", False),
                "openWorldHint": False,
            },
        }
        for name, spec in TOOLS.items()
    ]


def call_tool(clients: Clients, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        return {"content": [{"type": "text", "text": f"Unknown tool {name!r}."}], "isError": True}
    try:
        for key in spec["inputSchema"].get("required", []):
            if key not in arguments:
                raise ToolError(f"Missing required argument {key!r}.")
        payload = spec["fn"](clients, arguments)
    except ToolError as err:
        return {"content": [{"type": "text", "text": str(err)}], "isError": True}
    except (ValueError, TypeError, KeyError) as err:
        return {"content": [{"type": "text", "text": f"Bad arguments: {err}"}], "isError": True}
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


def handle(message: dict[str, Any], clients: Clients | None) -> dict[str, Any] | None:
    """One JSON-RPC message in, one reply out (None for notifications)."""
    method = message.get("method", "")
    msg_id = message.get("id")
    if msg_id is None:
        return None  # notifications/initialized, cancelled, etc.

    def ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def fail(code: int, text: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}}

    params = message.get("params") or {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok(
            {
                "protocolVersion": asked
                if asked in SUPPORTED_PROTOCOLS
                else SUPPORTED_PROTOCOLS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        )
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": list_tools()})
    if method == "tools/call":
        if clients is None:
            return ok(
                {
                    "content": [{"type": "text", "text": "Server is not configured."}],
                    "isError": True,
                }
            )
        return ok(call_tool(clients, params.get("name", ""), params.get("arguments") or {}))
    return fail(-32601, f"Method not found: {method}")


def main() -> int:
    try:
        clients: Clients | None = Clients(Config.from_env())
        startup_error = None
    except ToolError as err:
        clients, startup_error = None, str(err)
    if startup_error:
        print(f"yarnotron-mcp: {startup_error}", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        reply = handle(message, clients)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
