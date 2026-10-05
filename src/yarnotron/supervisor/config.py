"""Supervisor configuration, read from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Probe:
    """How to tell a service is up *and answering*, not merely listening."""

    kind: str  # "http" | "wyoming" | "tcp"
    url: str | None = None  # http
    host: str = "127.0.0.1"  # wyoming / tcp
    port: int | None = None
    expect_status: int = 200
    timeout: float = 3.0


@dataclass(frozen=True)
class PreStop:
    """A request made to a service before it is stopped, so it can save its work."""

    url: str
    method: str = "POST"
    timeout: float = 45.0
    bearer_token: str | None = None


@dataclass(frozen=True)
class Service:
    name: str
    label: str
    start: tuple[str, ...]
    probe: Probe
    port: int | None = None  # used to find an already-running process (lsof)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    start_timeout: float = 60.0
    stop_timeout: float = 15.0
    stop_policy: str = "term"  # "term" stops it on shutdown; "leave" never does
    process_match: str | None = None  # substring of the process name we are willing to restart
    pre_stop: PreStop | None = None
    restart_unresponsive: bool = True


@dataclass(frozen=True)
class SupervisorConfig:
    host: str
    port: int
    token: str | None
    state_dir: Path
    services: tuple[Service, ...]

    def service(self, name: str) -> Service:
        for service in self.services:
            if service.name == name:
                return service
        raise KeyError(name)

    def start_order(self) -> list[Service]:
        """Dependencies first. Raises ConfigError on a cycle or unknown name."""
        order: list[Service] = []
        seen: set[str] = set()
        visiting: set[str] = set()
        names = {s.name for s in self.services}

        def visit(service: Service) -> None:
            if service.name in seen:
                return
            if service.name in visiting:
                raise ConfigError(f"Service dependencies form a loop at {service.name!r}.")
            visiting.add(service.name)
            for dep in service.depends_on:
                if dep not in names:
                    raise ConfigError(f"{service.name!r} depends on unknown service {dep!r}.")
                visit(self.service(dep))
            visiting.discard(service.name)
            seen.add(service.name)
            order.append(service)

        for service in self.services:
            visit(service)
        return order


def _probe(raw: dict, where: str) -> Probe:
    kind = raw.get("type")
    if kind not in {"http", "wyoming", "tcp"}:
        raise ConfigError(f"{where}: probe type must be http, wyoming or tcp (got {kind!r}).")
    if kind == "http" and not raw.get("url"):
        raise ConfigError(f"{where}: an http probe needs a url.")
    if kind != "http" and not raw.get("port"):
        raise ConfigError(f"{where}: a {kind} probe needs a port.")
    return Probe(
        kind=kind,
        url=raw.get("url"),
        host=raw.get("host", "127.0.0.1"),
        port=raw.get("port"),
        expect_status=int(raw.get("expect_status", 200)),
        timeout=float(raw.get("timeout", 3.0)),
    )


def parse(data: dict) -> SupervisorConfig:
    section = data.get("supervisor", {})
    services: list[Service] = []
    for raw in data.get("service", []):
        name = raw.get("name")
        if not name:
            raise ConfigError("Every [[service]] needs a name.")
        where = f"service {name!r}"
        start = raw.get("start")
        if not start or not isinstance(start, list):
            raise ConfigError(f'{where}: start must be a list like ["ollama", "serve"].')
        if "probe" not in raw:
            raise ConfigError(f"{where}: needs a [service.probe] table.")
        pre = raw.get("pre_stop")
        services.append(
            Service(
                name=name,
                label=raw.get("label", name),
                start=tuple(str(part) for part in start),
                probe=_probe(raw["probe"], where),
                port=raw.get("port"),
                cwd=raw.get("cwd"),
                env={str(k): str(v) for k, v in raw.get("env", {}).items()},
                depends_on=tuple(raw.get("depends_on", [])),
                start_timeout=float(raw.get("start_timeout", 60)),
                stop_timeout=float(raw.get("stop_timeout", 15)),
                stop_policy=raw.get("stop_policy", "term"),
                process_match=raw.get("process_match"),
                pre_stop=PreStop(**pre) if pre else None,
                restart_unresponsive=bool(raw.get("restart_unresponsive", True)),
            )
        )
        if services[-1].stop_policy not in {"term", "leave"}:
            raise ConfigError(f'{where}: stop_policy must be "term" or "leave".')
    if not services:
        raise ConfigError("No [[service]] entries: nothing to supervise.")
    config = SupervisorConfig(
        host=section.get("host", "0.0.0.0"),
        port=int(section.get("port", 8990)),
        token=section.get("token") or None,
        state_dir=Path(section.get("state_dir", "~/.yarnotron-supervisor")).expanduser(),
        services=tuple(services),
    )
    config.start_order()  # validates dependencies
    return config


def load(path: str | Path) -> SupervisorConfig:
    path = Path(path).expanduser()
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc
    return parse(data)
