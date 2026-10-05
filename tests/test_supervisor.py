"""The Mac supervisor, against real throwaway processes (no mocks of the OS)."""

from __future__ import annotations

import json
import shutil
import socket
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from yarnotron.supervisor import procs
from yarnotron.supervisor.config import ConfigError, parse
from yarnotron.supervisor.core import Supervisor
from yarnotron.supervisor.probes import probe_http, probe_tcp, probe_wyoming
from yarnotron.supervisor.server import make_server

pytestmark = pytest.mark.skipif(not shutil.which("lsof"), reason="needs lsof")

# A tiny stand-in service. MODE controls how it misbehaves.
FAKE = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, mode = int(sys.argv[1]), sys.argv[2]
    if mode == "exit":
        print("boom: model file missing", flush=True); sys.exit(3)
    if mode == "silent":
        import socket
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port)); s.listen(); time.sleep(300)
    if mode == "never":
        time.sleep(300)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _j(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_GET(self): self._j({"ok": True})
        def do_POST(self):
            if mode == "prepare500": return self._j({"detail": "nope"}, 500)
            drained = mode != "busy"
            self._j({"drained": drained, "notes": [] if drained else ["1 turn still running"],
                     "story": {"title": "The Brave Fox", "scenes": 4}})
    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def service(name, port, mode="ok", **extra):
    raw = {
        "name": name,
        "label": name.title(),
        "start": [sys.executable, "-c", FAKE, str(port), mode],
        "port": port,
        "start_timeout": 6,
        "stop_timeout": 5,
        "probe": {"type": "http", "url": f"http://127.0.0.1:{port}/health", "timeout": 1},
    }
    raw.update(extra)
    return raw


@pytest.fixture
def make(tmp_path):
    supervisors = []

    def build(*services, token=None):
        config = parse(
            {
                "supervisor": {"port": 0, "state_dir": str(tmp_path), "token": token},
                "service": list(services),
            }
        )
        supervisor = Supervisor(config)
        supervisors.append(supervisor)
        return supervisor

    yield build
    for supervisor in supervisors:  # never leave a fake service behind
        for popen in supervisor._popen.values():
            if popen.poll() is None:
                popen.kill()
        for svc in supervisor.config.services:
            for pid in procs.listening_pids(svc.port or 0):
                procs.stop(pid, 1)


def run(supervisor, kind, **options):
    job = supervisor.start_job(kind, **options)
    deadline = time.monotonic() + 40
    while job.state != "done" and time.monotonic() < deadline:
        time.sleep(0.1)
    assert job.state == "done", "job never finished"
    return job


def outcomes(job):
    return {step.service: step.outcome for step in job.steps}


# ------------------------------------------------------------------- config


def test_config_rejects_dependency_loops_and_unknown_names():
    a = service("a", 1, depends_on=["b"])
    b = service("b", 2, depends_on=["a"])
    with pytest.raises(ConfigError, match="loop"):
        parse({"service": [a, b]})
    with pytest.raises(ConfigError, match="unknown service"):
        parse({"service": [service("a", 1, depends_on=["ghost"])]})
    with pytest.raises(ConfigError, match="nothing to supervise"):
        parse({})


# ------------------------------------------------------------------- probes


def test_probes_tell_down_from_unresponsive():
    port = free_port()
    assert probe_tcp("127.0.0.1", port, 1).state == "down"
    assert probe_http(f"http://127.0.0.1:{port}/", 200, 1).state == "down"

    held = socket.socket()  # listens but never answers
    held.bind(("127.0.0.1", 0))
    held.listen()
    try:
        silent = held.getsockname()[1]
        assert probe_tcp("127.0.0.1", silent, 1).ok
        assert probe_wyoming("127.0.0.1", silent, 0.5).state == "unresponsive"
    finally:
        held.close()


def test_wyoming_probe_accepts_an_info_reply():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()

    def answer():
        conn, _ = server.accept()
        conn.recv(100)
        conn.sendall(b'{"type": "info", "data_length": 0}\n')
        conn.close()

    threading.Thread(target=answer, daemon=True).start()
    result = probe_wyoming("127.0.0.1", server.getsockname()[1], 2)
    server.close()
    assert result.ok


# ------------------------------------------------------------------- launch


def test_status_reports_everything_down_when_nothing_runs(make):
    supervisor = make(service("alpha", free_port()))
    status = supervisor.status()
    assert status["ok"] is False
    assert status["services"][0]["state"] == "down"


def test_launch_starts_in_dependency_order_then_is_idempotent(make):
    p1, p2 = free_port(), free_port()
    supervisor = make(
        service("backend", p2, depends_on=["model"]),
        service("model", p1),
    )
    first = run(supervisor, "launch")
    assert first.ok, first.summary
    assert [step.service for step in first.steps] == ["model", "backend"]
    assert outcomes(first) == {"model": "started", "backend": "started"}
    assert supervisor.status()["ok"] is True

    second = run(supervisor, "launch")
    assert second.ok
    assert set(outcomes(second).values()) == {"already_up"}
    assert "already running" in second.summary


def test_launch_explains_a_missing_program_and_skips_dependents(make):
    p1, p2 = free_port(), free_port()
    broken = service("model", p1, start=["/no/such/ollama", "serve"])
    supervisor = make(broken, service("backend", p2, depends_on=["model"]))
    job = run(supervisor, "launch")

    assert not job.ok
    model, backend = job.steps
    assert model.issue.code == "command_not_found"
    assert "/no/such/ollama" in model.issue.detail
    assert backend.issue.code == "dependency_failed"
    assert supervisor.status()["services"][1]["state"] == "down"  # never started


def test_launch_shows_the_log_when_a_process_exits_immediately(make):
    crashy = {**service("crashy", free_port()), "start": [sys.executable, "-c", FAKE, "1", "exit"]}
    supervisor = make(crashy)
    job = run(supervisor, "launch")
    issue = job.steps[0].issue
    assert issue.code == "exited_early"
    assert "exit code 3" in issue.summary
    assert "boom: model file missing" in issue.detail


def test_launch_reports_a_process_that_never_answers(make):
    port = free_port()
    never = {
        **service("slow", port, start_timeout=1.5),
        "start": [sys.executable, "-c", FAKE, str(port), "never"],
    }
    supervisor = make(never)
    job = run(supervisor, "launch")
    issue = job.steps[0].issue
    assert issue.code == "start_timeout"
    assert "still running" in issue.detail
    assert "refused" in issue.detail


def test_launch_replaces_a_hung_copy_holding_the_port(make):
    port = free_port()
    hung = procs.spawn(
        (sys.executable, "-c", FAKE, str(port), "silent"),
        cwd=None,
        env={},
        log_path=Path("/dev/null"),
    )
    time.sleep(0.5)
    assert procs.listening_pids(port) == [hung.pid]
    supervisor = make(service("alpha", port, process_match="python"))
    job = run(supervisor, "launch")
    assert job.ok, job.steps[0].issue and job.steps[0].issue.detail
    step = job.steps[0]
    assert step.outcome == "started"
    assert any("not answering" in w for w in step.warnings)
    hung.wait(timeout=5)


def test_launch_will_not_kill_a_stranger_on_the_port(make):
    port = free_port()
    stranger = procs.spawn(
        (sys.executable, "-c", FAKE, str(port), "silent"),
        cwd=None,
        env={},
        log_path=Path("/dev/null"),
    )
    time.sleep(0.5)
    try:
        supervisor = make(service("alpha", port, process_match="definitely-not-this"))
        job = run(supervisor, "launch")
        assert job.steps[0].issue.code == "port_in_use"
        assert stranger.poll() is None  # untouched
    finally:
        stranger.kill()


# ----------------------------------------------------------------- shutdown


def stop_config(port, mode="ok", **extra):
    return service(
        "backend",
        port,
        mode,
        pre_stop={"url": f"http://127.0.0.1:{port}/prepare", "timeout": 5},
        **extra,
    )


def test_shutdown_saves_then_stops_and_reports_the_story(make):
    p1, p2 = free_port(), free_port()
    supervisor = make(stop_config(p2, depends_on=["model"]), service("model", p1))
    assert run(supervisor, "launch").ok

    job = run(supervisor, "shutdown")
    assert job.ok, job.summary
    assert [s.service for s in job.steps] == ["backend", "model"]  # reverse order
    assert outcomes(job) == {"backend": "stopped", "model": "stopped"}
    assert job.story == {"title": "The Brave Fox", "scenes": 4}
    assert job.summary.startswith("Saved The Brave Fox (4 scenes).")
    assert supervisor.status()["services"][0]["state"] == "down"


def test_shutdown_with_nothing_running_is_a_calm_no_op(make):
    supervisor = make(service("alpha", free_port()))
    job = run(supervisor, "shutdown")
    assert job.ok
    assert outcomes(job) == {"alpha": "already_stopped"}
    assert "Nothing was running" in job.summary


def test_shutdown_refuses_to_stop_a_backend_that_will_not_save(make):
    port = free_port()
    config = {**stop_config(port), "start": [sys.executable, "-c", FAKE, str(port), "prepare500"]}
    supervisor = make(config)
    assert run(supervisor, "launch").ok

    job = run(supervisor, "shutdown")
    assert not job.ok
    assert job.steps[0].issue.code == "prepare_failed"
    assert "HTTP 500" in job.steps[0].issue.summary
    assert supervisor.status()["ok"] is True  # still running, work not thrown away

    forced = run(supervisor, "shutdown", force=True)
    assert forced.ok
    assert any("Forced" in w for w in forced.steps[0].warnings)


def test_shutdown_keeps_the_model_server_up_when_the_backend_is_busy(make):
    p1, p2 = free_port(), free_port()
    busy = {
        **stop_config(p2, depends_on=["model"]),
        "start": [sys.executable, "-c", FAKE, str(p2), "busy"],
    }
    supervisor = make(busy, service("model", p1))
    assert run(supervisor, "launch").ok

    job = run(supervisor, "shutdown")
    backend, model = job.steps
    assert backend.issue.code == "not_drained"
    assert "1 turn still running" in backend.issue.detail
    assert model.issue.code == "kept_running"
    assert supervisor.status()["ok"] is True


def test_stop_policy_leave_is_respected(make):
    port = free_port()
    supervisor = make(service("model", port, stop_policy="leave"))
    assert run(supervisor, "launch").ok
    job = run(supervisor, "shutdown")
    assert outcomes(job) == {"model": "left_running"}
    assert supervisor.status()["ok"] is True


# --------------------------------------------------------------------- HTTP


def call(base, path, method="GET", token=None):
    request = urllib.request.Request(
        base + path, method=method, data=b"" if method == "POST" else None
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_api_requires_the_token_and_runs_jobs(make, tmp_path):
    supervisor = make(service("alpha", free_port()), token="s3cret")
    supervisor.config = supervisor.config.__class__(
        host="127.0.0.1",
        port=0,
        token="s3cret",
        state_dir=supervisor.config.state_dir,
        services=supervisor.config.services,
    )
    server = make_server(supervisor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert call(base, "/v1/health")[1]["auth_required"] is True  # open on purpose
        assert call(base, "/v1/status")[0] == 401
        assert call(base, "/v1/status", token="wrong")[0] == 401

        code, job = call(base, "/v1/launch", "POST", "s3cret")
        assert code == 202
        for _ in range(100):
            _, job = call(base, f"/v1/jobs/{job['id']}", token="s3cret")
            if job["state"] == "done":
                break
            time.sleep(0.2)
        assert job["ok"] is True
        assert call(base, "/v1/status", token="s3cret")[1]["ok"] is True
        assert call(base, "/v1/jobs/nope", token="s3cret")[0] == 404
        call(base, "/v1/shutdown", "POST", "s3cret")
    finally:
        server.shutdown()
        server.server_close()
