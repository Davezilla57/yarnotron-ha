"""An in-memory stand-in for the Yarnotron backend, served over real HTTP.

It speaks only the routes the integration uses. Tests script its behaviour
through plain attributes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from aiohttp import web

TOKEN = "secret-token"


class FakeBackend:
    def __init__(self) -> None:
        self.token: str | None = TOKEN
        self.down = False
        self.status_404 = False
        self.sessions_created = 0
        self.session_counter = 0
        self.ended: list[str] = []
        self.turns: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.reviewed: list[tuple[str, dict]] = []
        self.released: list[tuple[str, dict]] = []
        self.latest_session_id: str | None = None
        self.latest_session_ended = False
        self.strikes = 0
        self.stories: list[dict[str, Any]] = [{"id": "story_1", "title": "The Castle"}]
        self.ollama = True
        # What the next streamed turn does.
        self.sentences: list[str] = ["Once upon a time.", "A dragon yawned."]
        self.turn: dict[str, Any] = {
            "response_text": "Once upon a time. A dragon yawned.",
            "safety": {"action": "allow", "categories": []},
            "updated_story": {"id": "story_1"},
        }
        self.stream_error: dict[str, Any] | None = None
        self.create_session_404_for_story = False
        self.shutting_down = False
        self.prepare_calls = 0
        self.cancel_calls = 0

    # ----------------------------------------------------------------- helpers

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self._middleware])
        r = app.router
        r.add_get("/health", self.health)
        r.add_get("/v1/status", self.status)
        r.add_get("/v1/safety-events", self.safety_events)
        r.add_get("/v1/stories", self.list_stories)
        r.add_get("/v1/stories/{story_id}/export.md", self.export)
        r.add_post("/v1/sessions", self.create_session)
        r.add_post("/v1/sessions/{session_id}/end", self.end_session)
        r.add_post("/v1/safety-events/{event_id}/review", self.review)
        r.add_post("/v1/safety-events/{event_id}/release", self.release)
        r.add_post("/v1/agent/turn/stream", self.stream)
        r.add_post("/v1/system/prepare-shutdown", self.prepare_shutdown)
        r.add_post("/v1/system/cancel-shutdown", self.cancel_shutdown)
        return app

    @web.middleware
    async def _middleware(self, request: web.Request, handler):
        if self.down:
            raise web.HTTPServiceUnavailable()
        guarded = request.path.startswith("/v1/") and self.token is not None
        if guarded and request.headers.get("Authorization") != f"Bearer {self.token}":
            raise web.HTTPUnauthorized()
        return await handler(request)

    def add_event(
        self,
        event_id: str,
        *,
        action: str = "redirect",
        categories: list[str] | None = None,
        child_id: str = "local_child",
        created_at: datetime | None = None,
        transcript: str = "something scary",
        parent_note: str = "A note for parents",
    ) -> dict[str, Any]:
        event = {
            "id": event_id,
            "child_id": child_id,
            "story_id": "story_1",
            "session_id": self.latest_session_id,
            "safety": {
                "action": action,
                "severity": "medium",
                "categories": categories or [],
                "parent_note": parent_note,
            },
            "transcript_excerpt": transcript,
            "counted_as_strike": True,
            "reviewed_at": None,
            "created_at": (created_at or datetime.now(UTC)).isoformat(),
        }
        self.events.insert(0, event)  # newest first, like the backend
        return event

    # ---------------------------------------------------------------- handlers

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def status(self, request: web.Request) -> web.Response:
        if self.status_404:
            raise web.HTTPNotFound()
        return web.json_response(
            {
                "app": "yarnotron",
                "version": "test",
                "auth_required": self.token is not None,
                "strike_limit": 5,
                "shutting_down": self.shutting_down,
                "stories": len(self.stories),
                "models": {
                    "ollama_reachable": self.ollama,
                    "chat_model": "chat",
                    "safety_model": "safe",
                    "embedding_model": "embed",
                    "missing": [],
                },
                "child": {
                    "child_id": request.query.get("child_id"),
                    "open_sessions": 0 if self.latest_session_ended else 1,
                    "unreviewed_safety_events": sum(
                        1 for e in self.events if e["reviewed_at"] is None
                    ),
                    "pending_media": 0,
                    "latest_session_id": self.latest_session_id,
                    "latest_session_ended": self.latest_session_ended,
                    "latest_session_strikes": self.strikes,
                },
            }
        )

    async def prepare_shutdown(self, request: web.Request) -> web.Response:
        self.prepare_calls += 1
        self.shutting_down = True
        return web.json_response(
            {
                "draining": True,
                "drained": True,
                "sessions_ended": ["sess_1"],
                "story": {"story_id": "story_1", "title": "The Brave Fox", "scenes": 4},
                "checkpoint": "ok",
                "notes": [],
            }
        )

    async def cancel_shutdown(self, request: web.Request) -> web.Response:
        self.cancel_calls += 1
        self.shutting_down = False
        return web.json_response({"draining": False})

    async def safety_events(self, request: web.Request) -> web.Response:
        return web.json_response(self.events[: int(request.query.get("limit", 50))])

    async def list_stories(self, request: web.Request) -> web.Response:
        return web.json_response(self.stories)

    async def export(self, request: web.Request) -> web.Response:
        return web.Response(text=f"# {request.match_info['story_id']}\n", content_type="text/plain")

    async def create_session(self, request: web.Request) -> web.Response:
        body = await request.json()
        if self.create_session_404_for_story and body.get("story_id"):
            raise web.HTTPNotFound(text=json.dumps({"detail": "Story not found"}))
        self.session_counter += 1
        self.sessions_created += 1
        self.latest_session_id = f"sess_{self.session_counter}"
        self.latest_session_ended = False
        self.strikes = 0
        return web.json_response({"id": self.latest_session_id}, status=201)

    async def end_session(self, request: web.Request) -> web.Response:
        self.ended.append(request.match_info["session_id"])
        return web.json_response({"id": request.match_info["session_id"]})

    async def review(self, request: web.Request) -> web.Response:
        self.reviewed.append((request.match_info["event_id"], await request.json()))
        return web.json_response({"id": request.match_info["event_id"]})

    async def release(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.released.append((request.match_info["event_id"], body))
        if request.match_info["event_id"] == "gone":
            raise web.HTTPNotFound(text=json.dumps({"detail": "Safety event not found"}))
        return web.json_response({"id": request.match_info["event_id"]})

    async def stream(self, request: web.Request) -> web.StreamResponse:
        self.turns.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        for sentence in self.sentences:
            await response.write((json.dumps({"sentence": sentence}) + "\n").encode())
        if self.stream_error is not None:
            await response.write((json.dumps(self.stream_error) + "\n").encode())
        else:
            await response.write((json.dumps({"turn": self.turn}) + "\n").encode())
        await response.write_eof()
        return response
