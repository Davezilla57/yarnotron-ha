"""The Hermes MCP server, against stdlib fake servers and over real stdio."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "yarnotron_mcp.py"
spec = importlib.util.spec_from_file_location("yarnotron_mcp", MODULE)
mcp = importlib.util.module_from_spec(spec)
sys.modules["yarnotron_mcp"] = mcp
spec.loader.exec_module(mcp)

SECRET_WORDS = "the monster under my bed"
PARENT_NOTE = "A note for parents"


class Fake:
    """A scripted JSON server. routes: {(METHOD, path): fn(body, query, headers) -> (code, obj)}."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str, dict, dict]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self):
                path, _, query = self.path.partition("?")
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                fake.calls.append((self.command, path, body, dict(self.headers)))
                fn = fake.routes.get((self.command, path))
                code, obj = fn(body, query, self.headers) if fn else (404, {"detail": "nope"})
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _go

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def paths(self, method):
        return [p for m, p, _, _ in self.calls if m == method]


EVENTS = [
    {
        "id": f"evt_{i}",
        "child_id": "local_child",
        "story_id": "s1",
        "session_id": "sess",
        "safety": {
            "action": "redirect" if i % 2 else "block",
            "severity": "medium",
            "categories": ["gore"] if i % 2 else ["fear", "gore"],
            "parent_note": PARENT_NOTE,
        },
        "transcript_excerpt": SECRET_WORDS,
        "counted_as_strike": True,
        "reviewed_at": None if i < 3 else "2026-10-01T00:00:00Z",
        "created_at": "2026-10-01T00:00:00Z",
    }
    for i in range(1, 6)
]


def backend_routes(state):
    return {
        ("GET", "/health"): lambda *_: (200, {"status": "ok"}),
        ("GET", "/v1/status"): lambda *_: (
            200,
            {
                "version": "t",
                "shutting_down": state.get("draining", False),
                "stories": 2,
                "strike_limit": 5,
                "models": {"ollama_reachable": True},
                "child": {
                    "open_sessions": 1,
                    "unreviewed_safety_events": 3,
                    "pending_media": 0,
                    "latest_session_strikes": 1,
                },
            },
        ),
        ("GET", "/v1/safety-events"): lambda *_: (200, EVENTS),
        ("POST", "/v1/system/cancel-shutdown"): lambda *_: (200, {"draining": False}),
        ("GET", "/v1/stories"): lambda *_: (
            200,
            [
                {
                    "id": "s1",
                    "title": "The Castle " * 20,
                    "updated_at": "x",
                    "scenes": [{"text": SECRET_WORDS}],
                    "characters": [{"name": "Rex"}],
                }
            ],
        ),
        ("GET", "/v1/stories/s1/export.md"): lambda *_: (200, {}),
        **{
            ("POST", f"/v1/safety-events/evt_{i}/review"): lambda *_: (200, {}) for i in range(1, 6)
        },
    }


def job(kind, ok=True, story=None):
    steps = [
        {
            "service": "yarnotron",
            "label": "Yarnotron backend",
            "outcome": "stopped" if ok else "failed",
            "message": "m",
            "warnings": [],
            "issue": None if ok else {"code": "prepare_failed", "summary": "refused to save"},
        }
    ]
    return {
        "id": "j1",
        "kind": kind,
        "state": "done",
        "ok": ok,
        "summary": "Saved The Brave Fox (4 scenes). Stopped everything."
        if ok
        else "refused to save",
        "steps": steps,
        "story": story,
    }


def supervisor_routes(state):
    def start(kind):
        return lambda body, query, headers: (202, {**job(kind), "state": "running", "steps": []})

    state.setdefault("shutdown_queries", [])
    return {
        ("GET", "/v1/status"): lambda *_: (
            200,
            {
                "ok": False,
                "services": [
                    {
                        "name": "ollama",
                        "label": "Ollama",
                        "state": "up",
                        "ok": True,
                        "detail": "answering",
                    },
                    {
                        "name": "whisper",
                        "label": "Whisper",
                        "state": "down",
                        "ok": False,
                        "detail": "Nothing is listening (connection refused).",
                        "log": "/x/whisper.log",
                    },
                ],
            },
        ),
        ("POST", "/v1/launch"): start("launch"),
        ("POST", "/v1/shutdown"): lambda body, query, headers: (
            state["shutdown_queries"].append(query)
            or (202, {**job("shutdown"), "state": "running", "steps": []})
        ),
        ("GET", "/v1/jobs/j1"): lambda *_: (
            200,
            job("x", story={"title": "The Brave Fox", "scenes": 4}),
        ),
    }


@pytest.fixture
def world(monkeypatch):
    state: dict = {}
    backend = Fake(backend_routes(state))
    sup = Fake(supervisor_routes(state))
    monkeypatch.setattr(mcp.time, "sleep", lambda s: None)
    made = [backend, sup]

    def clients(**env):
        base = {
            "YARNOTRON_URL": backend.url,
            "YARNOTRON_API_TOKEN": "btok",
            "YARNOTRON_SUPERVISOR_URL": sup.url,
            "YARNOTRON_SUPERVISOR_TOKEN": "stok",
        }
        base.update(env)
        return mcp.Clients(mcp.Config.from_env(base))

    yield clients, backend, sup, state, made
    for f in made:
        f.close()


def call(clients, name, **args):
    result = mcp.call_tool(clients, name, args)
    text = result["content"][0]["text"]
    return result.get("isError", False), (json.loads(text) if not result.get("isError") else text)


# ----------------------------------------------------------------- protocol


def test_stdio_handshake_lists_tools_and_refuses_unconfigured_calls():
    proc = subprocess.run(
        [sys.executable, "-I", str(MODULE)],
        input="\n".join(
            json.dumps(m)
            for m in [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "status", "arguments": {}},
                },
                {"jsonrpc": "2.0", "id": 4, "method": "nope"},
            ]
        )
        + "\n",
        capture_output=True,
        text=True,
        timeout=20,
        env={"PATH": ""},
        check=False,
    )
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4]  # the notification got no reply
    assert replies[0]["result"]["serverInfo"]["name"] == "yarnotron"
    names = {t["name"] for t in replies[1]["result"]["tools"]}
    assert {"status", "launch", "shutdown"} <= names
    assert replies[2]["result"]["isError"] is True
    assert replies[3]["error"]["code"] == -32601
    assert "YARNOTRON_URL is not set" in proc.stderr


def test_there_is_no_tool_that_talks_to_the_child():
    assert not [n for n in mcp.TOOLS if "turn" in n or "say" in n or "speak" in n]


def test_destructive_tools_are_annotated():
    by_name = {t["name"]: t for t in mcp.list_tools()}
    assert by_name["shutdown"]["annotations"]["destructiveHint"] is True
    assert by_name["status"]["annotations"]["readOnlyHint"] is True


# ------------------------------------------------------------------- privacy


def test_safety_events_never_carry_the_childs_words(world):
    clients, *_ = world
    err, out = call(clients(), "safety_events", limit=10)
    dump = json.dumps(out)
    assert not err and len(out["events"]) == 5
    assert SECRET_WORDS not in dump and PARENT_NOTE not in dump
    assert out["events"][0]["categories"] == ["gore"]
    _, only = call(clients(), "safety_events", unreviewed_only=True)
    assert [e["id"] for e in only["events"]] == ["evt_1", "evt_2"]


def test_story_list_and_digest_carry_no_text(world):
    clients, *_ = world
    _, stories = call(clients(), "list_stories")
    assert SECRET_WORDS not in json.dumps(stories)
    assert len(stories["stories"][0]["title"]) <= 80 and stories["stories"][0]["scenes"] == 1
    _, digest = call(clients(), "digest")
    assert digest["events_in_window"] == 5 and digest["unreviewed"] == 2 and digest["strikes"] == 5
    assert digest["by_category"]["gore"] == 5 and SECRET_WORDS not in json.dumps(digest)


def test_story_text_is_off_unless_the_profile_allows_it(world):
    clients, *_ = world
    err, text = call(clients(), "story_text", story_id="s1", confirmed=True)
    assert err and "disabled for this profile" in text


# -------------------------------------------------------------- confirmations


def test_shutdown_needs_the_users_yes_and_force_needs_permission(world):
    clients, _, sup, *_ = world
    err, text = call(clients(), "shutdown")
    assert err and "Missing required argument 'confirmed'" in text
    err, text = call(clients(), "shutdown", confirmed=False)
    assert err and "explicit yes" in text
    err, text = call(clients(), "shutdown", confirmed=True, force=True)
    assert err and "Forced shutdown is disabled" in text
    assert sup.paths("POST") == []  # nothing reached the Mac


def test_release_is_disabled_by_default_and_confirmation_applies_when_enabled(world):
    clients, backend, *_ = world
    err, text = call(clients(), "release_event", event_id="evt_1", confirmed=True)
    assert err and "disabled for this profile" in text
    err, text = call(
        clients(YARNOTRON_MCP_ALLOW_RELEASE="1"),
        "release_event",
        event_id="evt_1",
        confirmed=False,
    )
    assert err and "explicit yes" in text
    assert backend.paths("POST") == []


def test_review_records_who_acted(world):
    clients, backend, *_ = world
    err, out = call(
        clients(YARNOTRON_MCP_ACTOR="hermes:yarnotron"),
        "review_event",
        event_id="evt_2",
        note="fine",
        confirmed=True,
    )
    assert not err and out["reviewed_by"] == "hermes:yarnotron"
    _, _, body, headers = backend.calls[-1]
    assert body == {"reviewed_by": "hermes:yarnotron", "review_note": "fine"}
    assert headers["Authorization"] == "Bearer btok"


# --------------------------------------------------------------- power paths


def test_direct_shutdown_reports_the_saved_story(world):
    clients, _, _, state, _ = world
    err, out = call(
        clients(YARNOTRON_MCP_ALLOW_FORCE="1"), "shutdown", confirmed=True, force=True
    )
    assert not err and out["via"] == "supervisor" and out["ok"] is True
    assert state["shutdown_queries"] == ["force=1"]
    assert "Home Assistant was not involved" in out["note"]


def test_direct_launch_resumes_a_backend_left_draining(world):
    clients, backend, _, state, _ = world
    state["draining"] = True
    err, out = call(clients(), "launch", wait_seconds=30)
    assert not err and out["ok"] is True
    assert "/v1/system/cancel-shutdown" in backend.paths("POST")


def test_status_without_home_assistant_merges_mac_and_backend(world):
    clients, *_ = world
    err, out = call(clients(), "status")
    assert not err and out["backend_answering"] is True and out["ok"] is False
    assert out["backend"]["unreviewed_safety_events"] == 3
    whisper = next(p for p in out["problems"] if p["where"] == "Whisper")
    assert "connection refused" in whisper["explanation"]


def test_status_explains_an_unreachable_backend(world):
    clients, *_ = world
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    err, out = call(clients(YARNOTRON_URL=f"http://127.0.0.1:{dead}"), "status")
    assert not err and out["backend_answering"] is False
    explanation = next(p for p in out["problems"] if p["where"] == "Yarnotron backend")[
        "explanation"
    ]
    assert "nothing is listening" in explanation


def test_wrong_token_is_named_as_such(world):
    clients, _, sup, *_ = world
    sup.routes[("GET", "/v1/status")] = lambda *_: (401, {"error": "unauthorized"})
    _, out = call(clients(), "status")
    assert any("rejected the token" in p["explanation"] for p in out["problems"])


def test_home_assistant_path_is_preferred_when_configured(world):
    clients, _, sup, _, made = world
    ha = Fake(
        {
            ("POST", "/api/services/yarnotron/status"): lambda *_: (
                200,
                {
                    "changed_states": [],
                    "service_response": {
                        "ok": False,
                        "summary": "On the Mac: Whisper not running.",
                        "components": [
                            {
                                "name": "Piper",
                                "side": "home_assistant",
                                "state": "ok",
                                "detail": "up",
                                "issue": None,
                            },
                            {
                                "name": "Whisper",
                                "side": "mac",
                                "state": "down",
                                "detail": "d",
                                "issue": {
                                    "summary": "Whisper is not running on the Mac.",
                                    "detail": "refused",
                                    "hints": ["launch"],
                                },
                            },
                        ],
                    },
                },
            ),
            ("POST", "/api/services/yarnotron/shutdown"): lambda body, *_: (
                200,
                {
                    "service_response": {
                        "ok": True,
                        "summary": "Saved The Brave Fox (4 scenes).",
                        "details": "md",
                        "kind": "shutdown",
                    }
                },
            ),
        }
    )
    made.append(ha)
    c = clients(HASS_URL=ha.url, HASS_TOKEN="hatok")
    _, out = call(c, "status")
    assert [x["name"] for x in out["components"]] == ["Piper", "Whisper"]
    assert out["problems"][0]["try"] == ["launch"]
    assert sup.calls == []  # the Mac was reached only through Home Assistant

    _, shut = call(c, "shutdown", confirmed=True)
    assert shut["via"] == "home_assistant" and shut["summary"].startswith("Saved The Brave Fox")
    _, path, body, headers = ha.calls[-1]
    assert body == {"force": False, "wait": True} and headers["Authorization"] == "Bearer hatok"
    assert "return_response" not in path  # query string is stripped from the logged path
    ha.close()
    made.remove(ha)
