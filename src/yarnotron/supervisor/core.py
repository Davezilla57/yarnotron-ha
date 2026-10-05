"""The supervisor itself: check, launch and shut down the configured services."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import procs
from .config import PreStop, Service, SupervisorConfig
from .probes import ProbeResult, run_probe


@dataclass
class Issue:
    """Something went differently than expected, explained in full."""

    service: str
    code: str
    summary: str  # one plain sentence, fit to be spoken
    detail: str  # what was observed
    hints: list[str] = field(default_factory=list)  # what to try


@dataclass
class Step:
    service: str
    label: str
    outcome: str  # already_up started stopped already_stopped left_running skipped failed
    message: str
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    issue: Issue | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Job:
    id: str
    kind: str  # "launch" | "shutdown"
    state: str = "running"  # running | done
    ok: bool = False
    summary: str = ""
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    story: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Supervisor:
    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self._popen: dict[str, object] = {}
        self._jobs: dict[str, Job] = {}
        self._busy = threading.Lock()

    # ------------------------------------------------------------------ status

    def _log(self, service: Service):
        return self.config.state_dir / "logs" / f"{service.name}.log"

    def _service_status(self, service: Service) -> dict:
        result = run_probe(service.probe)
        pids = procs.listening_pids(service.port) if service.port else []
        state = result.state
        if not result.ok and pids:
            state = "unresponsive"  # something holds the port but is not answering
        if result.ok:
            state = "up"
        return {
            "name": service.name,
            "label": service.label,
            "state": state,
            "ok": result.ok,
            "detail": result.detail,
            "latency_ms": result.latency_ms,
            "pids": pids,
            "process": procs.process_name(pids[0]) if pids else None,
            "log": str(self._log(service)),
        }

    def status(self) -> dict:
        order = self.config.start_order()
        with ThreadPoolExecutor(max_workers=len(order)) as pool:
            services = list(pool.map(self._service_status, order))
        running = [j.id for j in self._jobs.values() if j.state == "running"]
        return {
            "ok": all(s["ok"] for s in services),
            "services": services,
            "busy": bool(running),
            "running_job": running[0] if running else None,
            "time": time.time(),
        }

    # -------------------------------------------------------------------- jobs

    def job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def start_job(self, kind: str, **options) -> Job:
        """Begin a launch or shutdown in the background. One at a time."""
        for existing in self._jobs.values():
            if existing.state == "running":
                return existing  # asking twice joins the one in progress
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        self._jobs[job.id] = job
        work = self._run_launch if kind == "launch" else self._run_shutdown
        threading.Thread(target=work, args=(job,), kwargs=options, daemon=True).start()
        return job

    def _finish(self, job: Job, ok: bool, summary: str) -> None:
        job.ok = ok
        job.summary = summary
        job.finished_at = time.time()
        job.state = "done"

    # ------------------------------------------------------------------ launch

    def _run_launch(self, job: Job) -> None:
        with self._busy:
            failed: set[str] = set()
            try:
                for service in self.config.start_order():
                    blocked = [d for d in service.depends_on if d in failed]
                    if blocked:
                        failed.add(service.name)
                        job.steps.append(self._blocked_step(service, blocked))
                        continue
                    step = self._launch_one(service)
                    job.steps.append(step)
                    if step.outcome == "failed":
                        failed.add(service.name)
            except Exception as exc:  # noqa: BLE001 - never leave a job "running" forever
                self._finish(
                    job, False, f"The launch stopped unexpectedly: {type(exc).__name__}: {exc}"
                )
                return
        self._finish(job, not failed, self._launch_summary(job.steps))

    def _blocked_step(self, service: Service, blockers: list[str]) -> Step:
        names = ", ".join(self.config.service(b).label for b in blockers)
        issue = Issue(
            service.name,
            "dependency_failed",
            f"I didn't try to start {service.label} because {names} isn't running.",
            f"{service.label} needs {names}, which failed earlier in this launch.",
            [f"Fix {names} first (see its error), then launch again."],
        )
        return Step(service.name, service.label, "failed", issue.summary, issue=issue)

    def _launch_one(self, service: Service) -> Step:
        began = time.monotonic()

        def done(step: Step) -> Step:
            step.seconds = round(time.monotonic() - began, 1)
            return step

        first = run_probe(service.probe)
        if first.ok:
            return done(
                Step(service.name, service.label, "already_up", "Already running and answering.")
            )

        warnings: list[str] = []
        pids = procs.listening_pids(service.port) if service.port else []
        if pids:
            time.sleep(2)  # it may just be busy or still waking up
            again = run_probe(service.probe)
            if again.ok:
                return done(
                    Step(service.name, service.label, "already_up", "Answering after a short wait.")
                )
            command = procs.process_name(pids[0])
            matches = service.process_match is None or service.process_match in command
            if not (service.restart_unresponsive and matches):
                return done(
                    self._failed(
                        service,
                        Issue(
                            service.name,
                            "port_in_use",
                            f"Something else is using {service.label}'s port and it isn't answering.",
                            f"Port {service.port} is held by pid {pids[0]} ({command}), "
                            f"which does not respond like {service.label}: {again.detail}",
                            [
                                f"Stop that process (kill {pids[0]}) or change the port in the supervisor config.",
                                "If it is a leftover copy, the supervisor only restarts processes whose "
                                f"command contains {service.process_match!r}."
                                if service.process_match
                                else "Automatic restart of unresponsive processes is turned off for this service.",
                            ],
                        ),
                    )
                )
            outcome = procs.stop(pids[0], service.stop_timeout)
            warnings.append(
                f"{service.label} held its port but was not answering ({again.detail}); "
                f"I stopped it ({outcome}) and started a fresh copy."
            )

        try:
            popen = procs.spawn(
                service.start, cwd=service.cwd, env=service.env, log_path=self._log(service)
            )
        except FileNotFoundError as exc:
            return done(
                self._failed(
                    service,
                    Issue(
                        service.name,
                        "command_not_found",
                        f"I couldn't start {service.label}: the program wasn't found.",
                        f"Running {' '.join(service.start)!r} failed: {exc}. "
                        "launchd gives programs a short PATH, so Homebrew tools need a full path.",
                        [
                            f"Use the full path in `start` (run `which {service.start[0]}` in Terminal)."
                        ],
                    ),
                )
            )
        except OSError as exc:
            return done(
                self._failed(
                    service,
                    Issue(
                        service.name,
                        "spawn_failed",
                        f"I couldn't start {service.label}: the system refused.",
                        f"Running {' '.join(service.start)!r} failed: {exc}.",
                        ["Check the working directory exists and the program is executable."],
                    ),
                )
            )
        self._popen[service.name] = popen

        deadline = time.monotonic() + service.start_timeout
        last: ProbeResult = first
        while time.monotonic() < deadline:
            code = popen.poll()
            if code is not None:
                tail = procs.log_tail(self._log(service))
                return done(
                    self._failed(
                        service,
                        Issue(
                            service.name,
                            "exited_early",
                            f"{service.label} started and then quit straight away (exit code {code}).",
                            f"Command: {' '.join(service.start)}\nExit code: {code}\n"
                            f"Last log lines ({self._log(service)}):\n{tail or '(empty)'}",
                            [
                                "Run the command in Terminal to see the full error.",
                                "Common causes: a missing virtualenv or model, or a port already taken.",
                            ],
                        ),
                        warnings,
                    )
                )
            last = run_probe(service.probe)
            if last.ok:
                step = Step(
                    service.name,
                    service.label,
                    "started",
                    f"Started (pid {popen.pid}) and answering.",
                    warnings=warnings,
                )
                return done(step)
            time.sleep(0.5)

        tail = procs.log_tail(self._log(service))
        return done(
            self._failed(
                service,
                Issue(
                    service.name,
                    "start_timeout",
                    f"{service.label} started but isn't answering after {service.start_timeout:g} seconds.",
                    f"The process is still running (pid {popen.pid}) but the health check fails: "
                    f"{last.detail}\nLast log lines ({self._log(service)}):\n{tail or '(empty)'}",
                    [
                        "It may still be loading a model; try again in a minute.",
                        "Raise start_timeout for this service if it is simply slow.",
                    ],
                ),
                warnings,
            )
        )

    @staticmethod
    def _failed(service: Service, issue: Issue, warnings: list[str] | None = None) -> Step:
        return Step(
            service.name,
            service.label,
            "failed",
            issue.summary,
            warnings=warnings or [],
            issue=issue,
        )

    @staticmethod
    def _launch_summary(steps: list[Step]) -> str:
        failed = [s for s in steps if s.outcome == "failed"]
        if failed:
            first = failed[0]
            more = (
                f" and {len(failed) - 1} more problem{'s' if len(failed) > 2 else ''}"
                if len(failed) > 1
                else ""
            )
            return f"{first.issue.summary if first.issue else first.message}{more}"
        started = [s.label for s in steps if s.outcome == "started"]
        if not started:
            return "Everything on the Mac was already running."
        return f"Started {_join(started)}. Everything on the Mac is running."

    # ---------------------------------------------------------------- shutdown

    def _run_shutdown(self, job: Job, force: bool = False) -> None:
        with self._busy:
            blocked: set[str] = set()  # services we must not stop because a dependent is still up
            try:
                order = list(reversed(self.config.start_order()))
                for service in order:
                    if service.name in blocked:
                        job.steps.append(self._kept_step(service))
                        continue
                    step = self._stop_one(service, force, job)
                    job.steps.append(step)
                    if step.outcome == "failed":
                        blocked.update(service.depends_on)
            except Exception as exc:  # noqa: BLE001
                self._finish(
                    job, False, f"The shutdown stopped unexpectedly: {type(exc).__name__}: {exc}"
                )
                return
        failed = [s for s in job.steps if s.outcome == "failed"]
        self._finish(job, not failed, self._shutdown_summary(job, failed))

    def _kept_step(self, service: Service) -> Step:
        issue = Issue(
            service.name,
            "kept_running",
            f"I left {service.label} running because something that uses it could not be stopped.",
            f"Stopping {service.label} first would break the process that still depends on it.",
            ["Resolve the earlier error, then ask for a shutdown again."],
        )
        return Step(service.name, service.label, "failed", issue.summary, issue=issue)

    def _stop_one(self, service: Service, force: bool, job: Job) -> Step:
        began = time.monotonic()

        def done(step: Step) -> Step:
            step.seconds = round(time.monotonic() - began, 1)
            return step

        probe = run_probe(service.probe)
        pids = procs.listening_pids(service.port) if service.port else []
        popen = self._popen.get(service.name)
        if popen is not None and popen.poll() is None and popen.pid not in pids:
            pids.append(popen.pid)

        if service.stop_policy == "leave":
            return done(
                Step(
                    service.name,
                    service.label,
                    "left_running",
                    "Left running by configuration (stop_policy = leave).",
                )
            )
        if not probe.ok and not pids:
            return done(Step(service.name, service.label, "already_stopped", "Was not running."))

        warnings: list[str] = []
        extra: dict = {}
        if service.pre_stop and probe.ok:
            outcome = self._pre_stop(service, service.pre_stop)
            if outcome["error"] and not force:
                return done(self._failed(service, outcome["error"]))
            if outcome["error"] and force:
                warnings.append(f"Forced past a failed save step: {outcome['error'].summary}")
            body = outcome["body"] or {}
            if body and not body.get("drained", True) and not force:
                return done(
                    self._failed(
                        service,
                        Issue(
                            service.name,
                            "not_drained",
                            f"{service.label} was still busy, so I did not stop it.",
                            "The save step ended with work unfinished: "
                            f"{'; '.join(body.get('notes', [])) or json.dumps(body)}",
                            [
                                (
                                    "Wait for the story turn to finish and ask again, or force the "
                                    "shutdown (unfinished safety reviews will be lost)."
                                )
                            ],
                        ),
                    )
                )
            if body.get("story"):
                job.story = body["story"]
            extra["prepare"] = body
            warnings.extend(body.get("notes", []))

        if not pids:
            return done(
                self._failed(
                    service,
                    Issue(
                        service.name,
                        "no_pid",
                        f"{service.label} is running but I couldn't find its process to stop it.",
                        f"It answers ({probe.detail}), but nothing was found listening on port "
                        f"{service.port}. It may be on another machine, or `lsof` may be unavailable.",
                        [
                            "Stop it by hand, or set `port` for this service in the supervisor config."
                        ],
                    ),
                    warnings,
                )
            )

        results = [procs.stop(pid, service.stop_timeout) for pid in pids]
        if "killed" in results:
            warnings.append(
                f"{service.label} did not exit within {service.stop_timeout:g}s of a polite "
                "stop request, so it was force-killed."
            )
        after = run_probe(service.probe)
        if after.ok:
            return done(
                self._failed(
                    service,
                    Issue(
                        service.name,
                        "still_running",
                        f"{service.label} is still answering after I tried to stop it.",
                        f"Stopped pid(s) {pids} ({', '.join(results)}) but the health check still passes: "
                        f"{after.detail}. Something may be restarting it (a launchd job or the app).",
                        ["Quit the app that manages it, or unload its launchd job."],
                    ),
                    warnings,
                )
            )
        self._popen.pop(service.name, None)
        return done(
            Step(
                service.name,
                service.label,
                "stopped",
                "Stopped." if not warnings else "Stopped, with warnings.",
                warnings=warnings,
                extra=extra,
            )
        )

    def _pre_stop(self, service: Service, pre: PreStop) -> dict:
        request = urllib.request.Request(
            pre.url, method=pre.method, data=b"" if pre.method == "POST" else None
        )
        if pre.bearer_token:
            request.add_header("Authorization", f"Bearer {pre.bearer_token}")
        try:
            with urllib.request.urlopen(request, timeout=pre.timeout) as response:
                raw = response.read()
            body = json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode(errors="replace")[:300]
            return {
                "body": None,
                "error": Issue(
                    service.name,
                    "prepare_failed",
                    f"{service.label} refused the save-and-drain request (HTTP {exc.code}), so I didn't stop it.",
                    f"{pre.method} {pre.url} returned HTTP {exc.code}: {text}",
                    [
                        "HTTP 401 means the supervisor's api token doesn't match the backend's.",
                        "Ask for a forced shutdown only if you accept losing unsaved work.",
                    ],
                ),
            }
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {
                "body": None,
                "error": Issue(
                    service.name,
                    "prepare_failed",
                    f"I couldn't ask {service.label} to save its work, so I didn't stop it.",
                    f"{pre.method} {pre.url} failed within {pre.timeout:g}s: {exc}",
                    ["Check the backend is healthy, or force the shutdown."],
                ),
            }
        return {"body": body, "error": None}

    @staticmethod
    def _shutdown_summary(job: Job, failed: list[Step]) -> str:
        saved = ""
        if job.story:
            title = job.story.get("title", "the story")
            scenes = job.story.get("scenes")
            saved = f"Saved {title}" + (f" ({scenes} scenes). " if scenes is not None else ". ")
        if failed:
            first = failed[0]
            return saved + (first.issue.summary if first.issue else first.message)
        stopped = [s.label for s in job.steps if s.outcome == "stopped"]
        if not stopped:
            return saved + "Nothing was running on the Mac."
        return saved + f"Stopped {_join(stopped)}."


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]
