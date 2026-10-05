"""HTTP front for the supervisor. Token-authenticated; JSON in and out."""

from __future__ import annotations

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .core import Supervisor

log = logging.getLogger("yarnotron.supervisor")
API_VERSION = 1


def make_server(supervisor: Supervisor) -> ThreadingHTTPServer:
    config = supervisor.config

    class Handler(BaseHTTPRequestHandler):
        server_version = "YarnotronSupervisor"

        def log_message(self, fmt: str, *args) -> None:
            log.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            if not config.token:
                return True
            header = self.headers.get("Authorization", "")
            return header.startswith("Bearer ") and hmac.compare_digest(
                header[7:].encode(), config.token.encode()
            )

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            path = url.path.rstrip("/")
            if path == "/v1/health" and method == "GET":
                # Deliberately open: "is the supervisor reachable" is not secret.
                return self._send(
                    200, {"ok": True, "api": API_VERSION, "auth_required": bool(config.token)}
                )
            if not self._authorized():
                return self._send(
                    401, {"error": "unauthorized", "detail": "Missing or wrong bearer token."}
                )
            try:
                if path == "/v1/status" and method == "GET":
                    return self._send(200, supervisor.status())
                if path == "/v1/launch" and method == "POST":
                    return self._send(202, supervisor.start_job("launch").as_dict())
                if path == "/v1/shutdown" and method == "POST":
                    force = query.get("force", ["0"])[0] in {"1", "true", "yes"}
                    return self._send(202, supervisor.start_job("shutdown", force=force).as_dict())
                if path.startswith("/v1/jobs/") and method == "GET":
                    job = supervisor.job(path.rsplit("/", 1)[1])
                    if job is None:
                        return self._send(404, {"error": "not_found", "detail": "No such job."})
                    return self._send(200, job.as_dict())
            except Exception as exc:
                log.exception("request failed")
                return self._send(
                    500, {"error": "internal", "detail": f"{type(exc).__name__}: {exc}"}
                )
            return self._send(404, {"error": "not_found", "detail": f"{method} {path}"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    server = ThreadingHTTPServer((config.host, config.port), Handler)
    server.daemon_threads = True
    return server
