"""An in-memory stand-in for the Mac supervisor, served over real HTTP."""

from __future__ import annotations

from typing import Any

from aiohttp import web

SUP_TOKEN = "sup-secret"

LABELS = {"ollama": "Ollama", "whisper": "Whisper speech-to-text", "yarnotron": "Yarnotron backend"}


class FakeSupervisor:
    def __init__(self) -> None:
        self.token: str | None = SUP_TOKEN
        # name -> "up" | "down" | "unresponsive"
        self.states = dict.fromkeys(LABELS, "up")
        self.launch_fails: str | None = None  # service name that fails to start
        self.shutdown_fails = False
        self.drop_requests = 0  # close the next N connections without answering
        self.launch_calls = 0
        self.shutdown_calls: list[dict[str, str]] = []
        self._jobs: dict[str, dict[str, Any]] = {}
        self._polls: dict[str, int] = {}

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self._middleware])
        r = app.router
        r.add_get("/v1/health", self.health)
        r.add_get("/v1/status", self.status)
        r.add_post("/v1/launch", self.launch)
        r.add_post("/v1/shutdown", self.shutdown)
        r.add_get("/v1/jobs/{job_id}", self.job)
        return app

    @web.middleware
    async def _middleware(self, request: web.Request, handler):
        if self.drop_requests > 0:
            self.drop_requests -= 1
            request.transport.close()
            raise web.HTTPServiceUnavailable()
        guarded = request.path != "/v1/health" and self.token is not None
        if guarded and request.headers.get("Authorization") != f"Bearer {self.token}":
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    # ------------------------------------------------------------------ views

    def _services(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "label": LABELS[name],
                "state": state,
                "ok": state == "up",
                "detail": {
                    "up": "answering",
                    "down": "Nothing is listening (connection refused).",
                    "unresponsive": "Port held but not answering.",
                }[state],
                "latency_ms": 5 if state == "up" else None,
                "pids": [] if state == "down" else [4242],
                "process": None if state == "down" else "/opt/homebrew/bin/thing",
                "log": f"/Users/x/.yarnotron-supervisor/logs/{name}.log",
            }
            for name, state in self.states.items()
        ]

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "api": 1, "auth_required": self.token is not None})

    async def status(self, request: web.Request) -> web.Response:
        services = self._services()
        return web.json_response(
            {"ok": all(s["ok"] for s in services), "services": services, "busy": False}
        )

    # -------------------------------------------------------------------- jobs

    def _step(self, name: str, outcome: str, message: str, **extra: Any) -> dict[str, Any]:
        return {
            "service": name,
            "label": LABELS[name],
            "outcome": outcome,
            "message": message,
            "seconds": 0.1,
            "warnings": [],
            "issue": None,
            "extra": {},
            **extra,
        }

    async def launch(self, request: web.Request) -> web.Response:
        self.launch_calls += 1
        steps, ok, started = [], True, []
        for name in list(LABELS):  # already in dependency order
            if self.states[name] == "up":
                steps.append(self._step(name, "already_up", "Already running and answering."))
            elif self.launch_fails == name:
                ok = False
                issue = {
                    "service": name,
                    "code": "exited_early",
                    "summary": f"{LABELS[name]} started and then quit straight away (exit code 3).",
                    "detail": "Exit code: 3\nLast log lines:\nboom: model file missing",
                    "hints": ["Run the command in Terminal to see the full error."],
                }
                steps.append(self._step(name, "failed", issue["summary"], issue=issue))
                break
            else:
                self.states[name] = "up"
                started.append(LABELS[name])
                steps.append(self._step(name, "started", "Started and answering."))
        summary = (
            "Everything on the Mac was already running."
            if ok and not started
            else f"Started {' and '.join(started)}. Everything on the Mac is running."
            if ok
            else steps[-1]["issue"]["summary"]
        )
        return self._accept("launch", ok, summary, steps, None)

    async def shutdown(self, request: web.Request) -> web.Response:
        self.shutdown_calls.append(dict(request.query))
        if self.shutdown_fails and request.query.get("force") != "1":
            issue = {
                "service": "yarnotron",
                "code": "prepare_failed",
                "summary": "Yarnotron backend refused the save-and-drain request (HTTP 500), "
                "so I didn't stop it.",
                "detail": "POST http://127.0.0.1:8989/v1/system/prepare-shutdown returned HTTP 500",
                "hints": ["Ask for a forced shutdown only if you accept losing unsaved work."],
            }
            steps = [self._step("yarnotron", "failed", issue["summary"], issue=issue)]
            return self._accept("shutdown", False, issue["summary"], steps, None)
        steps = []
        for name in reversed(list(LABELS)):
            self.states[name] = "down"
            steps.append(self._step(name, "stopped", "Stopped."))
        story = {"title": "The Brave Fox", "scenes": 4}
        return self._accept(
            "shutdown", True, "Saved The Brave Fox (4 scenes). Stopped everything.", steps, story
        )

    def _accept(self, kind, ok, summary, steps, story) -> web.Response:
        job_id = f"job{len(self._jobs) + 1}"
        self._jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "state": "done",
            "ok": ok,
            "summary": summary,
            "steps": steps,
            "story": story,
        }
        self._polls[job_id] = 0
        return web.json_response(
            {**self._jobs[job_id], "state": "running", "steps": []}, status=202
        )

    async def job(self, request: web.Request) -> web.Response:
        job_id = request.match_info["job_id"]
        if job_id not in self._jobs:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(self._jobs[job_id])
