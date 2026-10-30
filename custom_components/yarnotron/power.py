"""Launching, shutting down and reporting on everything Yarnotron needs.

Two sides, one report:

* Home Assistant's side -- this integration, the voice agent, the link to the
  backend, the Wyoming speech services, the voice satellites, the story session.
* The Mac's side -- Ollama, Whisper and the Yarnotron backend, which a small
  supervisor on the Mac checks, starts and stops on our behalf.

Every component that is not as expected carries an *issue*: a sentence fit to be
spoken, what was observed, and what to try. The spoken reply stays short; the full
account goes to a persistent notification, the sensor attributes and the log.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .api import YarnotronAuthError, YarnotronError
from .const import (
    DOMAIN,
    EVENT_POWER_FINISHED,
    JOB_POLL_SECONDS,
    LAUNCH_JOB_TIMEOUT,
    SHUTDOWN_JOB_TIMEOUT,
    WAKE_POLL_SECONDS,
    WAKE_WAIT_SECONDS,
)
from .supervisor_client import (
    SupervisorAuthError,
    SupervisorClient,
    SupervisorError,
    SupervisorUnreachableError,
)
from .wol import send_magic_packet

if TYPE_CHECKING:
    from .models import YarnotronConfigEntry

_LOGGER = logging.getLogger(__name__)

OK, DOWN, PROBLEM, ATTENTION, UNKNOWN = "ok", "down", "problem", "attention", "unknown"
_WORDS = {
    OK: "running",
    DOWN: "not running",
    PROBLEM: "not responding",
    ATTENTION: "needs attention",
    UNKNOWN: "unknown",
}


@dataclass
class Issue:
    summary: str  # one sentence, fit to be spoken
    detail: str = ""  # what was observed
    hints: list[str] = field(default_factory=list)  # what to try


@dataclass
class Component:
    name: str
    side: str  # "home_assistant" | "mac"
    state: str
    detail: str = ""
    issue: Issue | None = None

    @property
    def spoken(self) -> str:
        return f"{self.name} {_WORDS[self.state]}"


@dataclass
class Report:
    components: list[Component]

    @property
    def problems(self) -> list[Component]:
        return [c for c in self.components if c.state in (DOWN, PROBLEM)]

    @property
    def ok(self) -> bool:
        return not self.problems

    def side(self, side: str) -> list[Component]:
        return [c for c in self.components if c.side == side]

    def spoken(self) -> str:
        parts = []
        for side, label in (("home_assistant", "On Home Assistant"), ("mac", "On the host computer")):
            items = self.side(side)
            if items:
                parts.append(f"{label}: " + ", ".join(c.spoken for c in items) + ".")
        flagged = [c for c in self.components if c.issue]
        if not flagged:
            parts.append("Everything is working.")
        else:
            parts.append(
                f"{len(flagged)} thing{'s need' if len(flagged) != 1 else ' needs'} a look. "
                "The details are in your Home Assistant notifications."
            )
        return " ".join(parts)

    def markdown(self) -> str:
        lines = ["| Component | Where | Status |", "|---|---|---|"]
        for c in self.components:
            where = "Home Assistant" if c.side == "home_assistant" else "Host computer"
            lines.append(f"| {c.name} | {where} | {_WORDS[c.state]} |")
        for c in self.components:
            if not c.issue:
                continue
            lines += ["", f"**{c.name}** — {c.issue.summary}"]
            if c.issue.detail:
                lines += ["", "```", c.issue.detail, "```"]
            lines += [f"- {hint}" for hint in c.issue.hints]
        return "\n".join(lines)

    def as_dict(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.components]


@dataclass
class PowerResult:
    kind: str
    ok: bool
    spoken: str
    markdown: str = ""
    job: dict[str, Any] | None = None


class PowerOperation:
    """A launch or shutdown running in the background, which a caller may wait on."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.result: PowerResult | None = None
        self.task: asyncio.Task | None = None

    async def wait(self, timeout: float) -> PowerResult | None:
        """The result, or None if it is still going after `timeout` seconds."""
        assert self.task is not None
        await asyncio.wait({self.task}, timeout=timeout)
        return self.result


class PowerController:
    def __init__(self, hass: HomeAssistant, entry: YarnotronConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._operation: PowerOperation | None = None

    # ---------------------------------------------------------------- helpers

    @property
    def _rt(self):
        return self.entry.runtime_data

    @property
    def supervisor(self) -> SupervisorClient | None:
        return self._rt.supervisor

    @property
    def last_operation(self) -> PowerOperation | None:
        return self._operation

    @property
    def busy(self) -> PowerOperation | None:
        op = self._operation
        return op if op and op.task and not op.task.done() else None

    # ----------------------------------------------------------------- status

    async def async_status(self) -> Report:
        mac = await self._mac_components()
        return Report([*self._ha_components(), *mac])

    def _ha_components(self) -> list[Component]:
        rt = self._rt
        out: list[Component] = []

        loaded = self.entry.state is ConfigEntryState.LOADED
        out.append(
            Component(
                "Yarnotron integration",
                "home_assistant",
                OK if loaded else PROBLEM,
                f"Config entry state: {self.entry.state}",
            )
        )

        registry = er.async_get(self.hass)
        agent_id = registry.async_get_entity_id("conversation", DOMAIN, self.entry.entry_id)
        agent_state = self.hass.states.get(agent_id) if agent_id else None
        if agent_state is None or agent_state.state == "unavailable":
            out.append(
                Component(
                    "Yarnotron voice agent",
                    "home_assistant",
                    PROBLEM,
                    "The conversation agent is not registered.",
                    Issue(
                        "The Yarnotron voice agent isn't available in Home Assistant.",
                        f"Entity {agent_id or '(none registered)'} state: "
                        f"{agent_state.state if agent_state else 'missing'}",
                        [
                            "Reload the Yarnotron integration (Settings > Devices & services).",
                            "Check Settings > System > Logs for errors from custom_components.yarnotron.",
                        ],
                    ),
                )
            )
        else:
            out.append(Component("Yarnotron voice agent", "home_assistant", OK, "Registered."))

        coordinator = rt.coordinator
        if coordinator.last_update_success:
            out.append(
                Component(
                    "Backend link",
                    "home_assistant",
                    OK,
                    f"Last poll of {rt.api.base_url} succeeded.",
                )
            )
        else:
            error = coordinator.last_exception
            kind = (
                "rejected the token" if isinstance(error, YarnotronAuthError) else "did not answer"
            )
            out.append(
                Component(
                    "Backend link",
                    "home_assistant",
                    DOWN,
                    f"{rt.api.base_url} {kind}.",
                    Issue(
                        "Home Assistant can't reach the Yarnotron backend."
                        if not isinstance(error, YarnotronAuthError)
                        else "The Yarnotron backend rejected Home Assistant's API token.",
                        f"Polling {rt.api.base_url} failed: {error or 'unknown error'}",
                        [
                            "Say 'launch Yarnotron' to start the host side."
                            if not isinstance(error, YarnotronAuthError)
                            else "Update the API token under the integration's Reconfigure option."
                        ],
                    ),
                )
            )

        out.extend(self._wyoming_components(registry))

        satellites = [e for e in registry.entities.values() if e.domain == "assist_satellite"]
        if satellites:
            states = [self.hass.states.get(e.entity_id) for e in satellites]
            up = [s for s in states if s is not None and s.state != "unavailable"]
            if up:
                out.append(
                    Component(
                        "Voice satellites",
                        "home_assistant",
                        OK,
                        f"{len(up)} of {len(satellites)} available.",
                    )
                )
            else:
                out.append(
                    Component(
                        "Voice satellites",
                        "home_assistant",
                        PROBLEM,
                        "All unavailable.",
                        Issue(
                            "No voice satellite is available.",
                            ", ".join(e.entity_id for e in satellites),
                            ["Check the satellite's power and Wi-Fi."],
                        ),
                    )
                )

        if rt.sessions.locked:
            out.append(
                Component(
                    "Story session",
                    "home_assistant",
                    ATTENTION,
                    "Locked after the safety strike limit; a grown-up must start a new one.",
                )
            )
        else:
            out.append(Component("Story session", "home_assistant", OK, "Not locked."))
        return out

    def _wyoming_components(self, registry: er.EntityRegistry) -> list[Component]:
        found = [
            e
            for e in registry.entities.values()
            if e.platform == "wyoming" and e.domain in ("stt", "tts")
        ]
        if not found:
            return [
                Component(
                    "Speech services (Wyoming)",
                    "home_assistant",
                    UNKNOWN,
                    "No Wyoming speech-to-text or text-to-speech entity found.",
                )
            ]
        out = []
        for entry in sorted(found, key=lambda e: e.entity_id):
            state = self.hass.states.get(entry.entity_id)
            name = entry.name or entry.original_name or entry.entity_id
            label = f"{name} ({'speech-to-text' if entry.domain == 'stt' else 'text-to-speech'})"
            if state is None or state.state == "unavailable":
                out.append(
                    Component(
                        label,
                        "home_assistant",
                        PROBLEM,
                        f"{entry.entity_id} is unavailable.",
                        Issue(
                            f"Home Assistant can't reach {name}.",
                            f"Entity {entry.entity_id} is "
                            f"{state.state if state else 'missing'}. The Wyoming server it points "
                            "to is not answering.",
                            [
                                "For Piper, restart the Piper add-on (Settings > Add-ons).",
                                "For Whisper on the host computer, launch Yarnotron so the supervisor starts it.",
                                "Check the Wyoming integration's host and port.",
                            ],
                        ),
                    )
                )
            else:
                out.append(Component(label, "home_assistant", OK, f"{entry.entity_id} available."))
        return out

    async def _mac_components(self) -> list[Component]:
        client = self.supervisor
        if client is None:
            # No supervisor is the normal case when the backend runs in a container: nothing to report on that side.
            return []
        try:
            status = await client.status()
        except SupervisorUnreachableError as err:
            return [self._unreachable_component(err)]
        except SupervisorAuthError as err:
            return [
                Component(
                    "Host supervisor",
                    "mac",
                    PROBLEM,
                    str(err),
                    Issue(
                        "The Host supervisor rejected Home Assistant's token.",
                        f"{client.base_url} answered 401.",
                        [
                            (
                                "Make the supervisor_token in the integration options match the "
                                "token in ~/.yarnotron-supervisor/supervisor.toml on the host computer."
                            )
                        ],
                    ),
                )
            ]
        except SupervisorError as err:
            return [
                Component(
                    "Host supervisor",
                    "mac",
                    PROBLEM,
                    str(err),
                    Issue(
                        "The Host supervisor gave an unexpected answer.",
                        str(err),
                        ["Check ~/.yarnotron-supervisor/logs/supervisor.log on the host computer."],
                    ),
                )
            ]

        out = [Component("Host supervisor", "mac", OK, f"Answering at {client.base_url}.")]
        for svc in status.get("services", []):
            state = {"up": OK, "down": DOWN, "unresponsive": PROBLEM}.get(svc.get("state"), UNKNOWN)
            issue = None
            if state != OK:
                issue = Issue(
                    f"{svc['label']} is {_WORDS[state]} on the host computer.",
                    f"{svc.get('detail', '')}"
                    + (
                        f"\nProcess: {svc['process']} (pid {svc['pids'][0]})"
                        if svc.get("pids")
                        else ""
                    )
                    + (f"\nLog: {svc['log']}" if svc.get("log") else ""),
                    [
                        "Say 'launch Yarnotron' to start anything that is down."
                        if state == DOWN
                        else "Say 'launch Yarnotron': a stuck copy is restarted automatically."
                    ],
                )
            out.append(Component(svc["label"], "mac", state, svc.get("detail", ""), issue))
        return out

    def _unreachable_component(self, err: SupervisorUnreachableError) -> Component:
        client = self.supervisor
        assert client is not None
        if err.kind == "refused":
            summary = "The host computer is on, but the Yarnotron supervisor isn't running."
            hints = [
                "On the host computer run: launchctl kickstart -k gui/$(id -u)/com.yarnotron.supervisor",
                "If that fails, run scripts/install_supervisor_launchd.sh again.",
                "The LaunchAgent only runs while you're logged in to the host computer.",
            ]
        elif err.kind == "timeout":
            summary = "The host computer isn't answering. It's probably asleep, off, or off the network."
            hints = [
                "Wake the Mac, or set its MAC address in the options so I can wake it for you.",
                "Keep it awake while it's on power: System Settings > Battery > Options.",
            ]
        else:
            summary = "I can't reach the host computer. It may be asleep or off, or the address may be wrong."
            hints = [
                "Check the supervisor address in the integration options.",
                "Check the host computer is awake and on the same network as Home Assistant.",
                "Keep it awake while it's on power: System Settings > Battery > Options.",
            ]
        return Component(
            "Host supervisor",
            "mac",
            DOWN,
            summary,
            Issue(
                summary, f"Request to {client.base_url} failed ({err.kind}): {err.detail}", hints
            ),
        )

    # ------------------------------------------------------------ operations

    def async_launch(self) -> PowerOperation:
        return self._begin("launch", self._launch())

    def async_shutdown(self, *, force: bool = False) -> PowerOperation:
        return self._begin("shutdown", self._shutdown(force))

    def _begin(self, kind: str, coro) -> PowerOperation:
        if (running := self.busy) is not None:
            coro.close()
            return running  # asking twice joins the one already in progress
        op = PowerOperation(kind)

        async def runner() -> None:
            try:
                op.result = await coro
            except Exception as err:
                _LOGGER.exception("Yarnotron %s failed unexpectedly", kind)
                op.result = PowerResult(
                    kind,
                    False,
                    f"Something unexpected went wrong during {kind}. I've logged the details.",
                    f"Unexpected error: `{type(err).__name__}: {err}`",
                )
            self._announce(op.result)

        self._operation = op
        op.task = self.entry.async_create_background_task(self.hass, runner(), f"yarnotron_{kind}")
        return op

    def _announce(self, result: PowerResult) -> None:
        notification_id = f"{DOMAIN}_power_{self.entry.entry_id}"
        if result.ok:
            persistent_notification.async_dismiss(self.hass, notification_id)
        else:
            persistent_notification.async_create(
                self.hass,
                result.markdown or result.spoken,
                title=f"Yarnotron {result.kind} needs attention",
                notification_id=notification_id,
            )
        self.hass.bus.async_fire(
            EVENT_POWER_FINISHED,
            {
                "kind": result.kind,
                "ok": result.ok,
                "summary": result.spoken,
                "entry_id": self.entry.entry_id,
            },
        )

    # ------------------------------------------------------------------ launch

    async def _launch(self) -> PowerResult:
        client = self.supervisor
        if client is None:
            return self._not_configured("launch")

        woke = False
        try:
            await client.health()
        except SupervisorUnreachableError as err:
            mac = self._rt.mac_address
            if mac is None:
                return self._unreachable_result("launch", err, tried_wake=False)
            woke = True
            try:
                await send_magic_packet(mac)
            except (OSError, ValueError) as wol_err:
                _LOGGER.warning("Could not send the wake packet: %s", wol_err)
            if not await self._wait_for_supervisor(client):
                return self._unreachable_result("launch", err, tried_wake=True)
        except SupervisorError as err:
            return self._supervisor_error("launch", err)

        try:
            job = await self._run_job(client, await client.launch(), LAUNCH_JOB_TIMEOUT)
        except SupervisorAuthError:
            return self._auth_result("launch")
        except (SupervisorUnreachableError, SupervisorError) as err:
            return self._supervisor_error("launch", err)

        await self._after_launch(job)
        report = await self.async_status()
        steps = job.get("steps", [])
        issues = [s for s in steps if s.get("issue")]
        ok = bool(job.get("ok")) and report.ok
        spoken = job.get("summary") or "Launch finished."
        if woke:
            spoken = "I woke the host computer. " + spoken
        if not ok:
            spoken += " The details are in your Home Assistant notifications."
        self._rt.powered_down = False if job.get("ok") else self._rt.powered_down
        md = [
            f"**Launch {'finished' if ok else 'did not complete'}.** {job.get('summary', '')}",
            "",
            self._steps_markdown(steps),
        ]
        if issues or not report.ok:
            md += ["", "**Status after launch**", "", report.markdown()]
        return PowerResult("launch", ok, spoken, "\n".join(md), job)

    async def _after_launch(self, job: dict[str, Any]) -> None:
        """Tell the backend to take turns again if an earlier shutdown left it draining."""
        if not job.get("ok"):
            return
        try:
            status = await self._rt.api.status(self._rt.coordinator.child_id)
            if status.get("shutting_down"):
                await self._rt.api.cancel_shutdown()
        except YarnotronError as err:
            _LOGGER.debug("Could not clear the shutdown flag: %s", err)
        await self._rt.coordinator.async_request_refresh()

    async def _wait_for_supervisor(self, client: SupervisorClient) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + WAKE_WAIT_SECONDS
        while loop.time() < deadline:
            try:
                await client.health()
                return True
            except SupervisorUnreachableError:
                await asyncio.sleep(WAKE_POLL_SECONDS)
            except SupervisorError:
                return True  # something answered
        return False

    # ---------------------------------------------------------------- shutdown

    async def _shutdown(self, force: bool) -> PowerResult:
        rt = self._rt
        client = self.supervisor
        # The child's side first: close the session locally, so nothing is left
        # pointing at a story that is about to stop being served.
        try:
            await rt.sessions.async_end_session()
        except YarnotronError as err:
            _LOGGER.info("Could not end the session before shutdown: %s", err)

        if client is None:
            return self._not_configured("shutdown")

        try:
            started = await client.shutdown(force=force)
        except SupervisorUnreachableError as err:
            return await self._shutdown_without_supervisor(err)
        except SupervisorAuthError:
            return self._auth_result("shutdown")
        except SupervisorError as err:
            return self._supervisor_error("shutdown", err)

        try:
            job = await self._run_job(client, started, SHUTDOWN_JOB_TIMEOUT)
        except (SupervisorUnreachableError, SupervisorError) as err:
            return self._supervisor_error("shutdown", err)

        ok = bool(job.get("ok"))
        if ok:
            rt.powered_down = True
        await rt.coordinator.async_request_refresh()
        spoken = job.get("summary") or "Shutdown finished."
        if not ok:
            spoken += " The details are in your Home Assistant notifications."
        md = [
            f"**Shutdown {'finished' if ok else 'did not complete'}.** {job.get('summary', '')}",
            "",
            self._steps_markdown(job.get("steps", [])),
        ]
        if not ok:
            md += [
                "",
                (
                    "Nothing was force-stopped while work was unsaved. "
                    "Say 'launch Yarnotron' to resume, or run the `yarnotron.shutdown` service "
                    "with force turned on to stop regardless."
                ),
            ]
        return PowerResult("shutdown", ok, spoken, "\n".join(md), job)

    async def _shutdown_without_supervisor(self, err: SupervisorUnreachableError) -> PowerResult:
        """The supervisor is out of reach. Save what we can and say what that means."""
        rt = self._rt
        try:
            report = await rt.api.prepare_shutdown()
        except YarnotronError:
            rt.powered_down = True
            return PowerResult(
                "shutdown",
                True,
                "I can't reach the host computer, and the backend isn't answering either, "
                "so there is nothing running to shut down.",
                f"The supervisor at {self.supervisor.base_url} is unreachable ({err.kind}: "
                f"{err.detail}) and the backend does not answer. Nothing is running.",
            )
        story = (report.get("story") or {}).get("title")
        saved = f"I saved {story}, " if story else "I saved your work, "
        spoken = (
            saved + "but I can't reach the Host supervisor, so the backend is still running "
            "and refusing new stories. The details are in your Home Assistant notifications."
        )
        comp = self._unreachable_component(err)
        md = (
            "The backend was told to finish and save, and it did. It could not be "
            "stopped because the supervisor is unreachable.\n\n"
            f"**{comp.issue.summary}**\n\n```\n{comp.issue.detail}\n```\n"
            + "\n".join(f"- {h}" for h in comp.issue.hints)
        )
        return PowerResult("shutdown", False, spoken, md)

    # ---------------------------------------------------------------- jobs

    async def _run_job(
        self, client: SupervisorClient, job: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while job.get("state") != "done":
            if loop.time() > deadline:
                job = {
                    **job,
                    "ok": False,
                    "state": "done",
                    "summary": f"This took longer than {timeout:g} seconds, so I stopped waiting.",
                }
                break
            await asyncio.sleep(JOB_POLL_SECONDS)
            job = await client.job(job["id"])
        return job

    @staticmethod
    def _steps_markdown(steps: list[dict[str, Any]]) -> str:
        lines = []
        for step in steps:
            icon = (
                "✅"
                if step["outcome"]
                in ("started", "stopped", "already_up", "already_stopped", "left_running")
                else "❌"
            )
            lines.append(f"- {icon} **{step['label']}**: {step['message']} ({step['seconds']}s)")
            for warning in step.get("warnings", []):
                lines.append(f"  - ⚠️ {warning}")
            issue = step.get("issue")
            if issue:
                lines += ["", f"  **What happened:** {issue['detail']}".replace("\n", "\n  "), ""]
                lines += [f"  - Try: {hint}" for hint in issue.get("hints", [])]
        return "\n".join(lines)

    # ------------------------------------------------------- failure wording

    def _not_configured(self, kind: str) -> PowerResult:
        text = (
            f"I can't {kind} the host side because no supervisor is set up. "
            "Add its address in the Yarnotron integration options."
        )
        return PowerResult(kind, False, text, text)

    def _auth_result(self, kind: str) -> PowerResult:
        text = f"The Host supervisor rejected my token, so I couldn't {kind}."
        md = (
            f"{text}\n\nMake the supervisor_token in the integration options match the token in "
            "`~/.yarnotron-supervisor/supervisor.toml` on the host computer."
        )
        return PowerResult(kind, False, text + " The details are in your notifications.", md)

    def _supervisor_error(self, kind: str, err: Exception) -> PowerResult:
        text = f"The Host supervisor stopped answering properly during {kind}."
        return PowerResult(
            kind,
            False,
            text + " The details are in your notifications.",
            f"{text}\n\n```\n{type(err).__name__}: {err}\n```\n"
            "Check `~/.yarnotron-supervisor/logs/supervisor.log` on the host computer.",
        )

    def _unreachable_result(
        self, kind: str, err: SupervisorUnreachableError, *, tried_wake: bool
    ) -> PowerResult:
        comp = self._unreachable_component(err)
        assert comp.issue is not None
        summary = comp.issue.summary
        if tried_wake:
            summary += f" I sent a wake-up signal and waited {WAKE_WAIT_SECONDS} seconds, but it didn't come up."
        md = f"**{summary}**\n\n```\n{comp.issue.detail}\n```\n" + "\n".join(
            f"- {h}" for h in comp.issue.hints
        )
        return PowerResult(kind, False, summary + " The details are in your notifications.", md)
