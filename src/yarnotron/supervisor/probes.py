"""Ask a service whether it is really answering."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Probe


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    state: str  # "up" | "down" | "unresponsive"
    detail: str
    latency_ms: int | None = None


def _timed(start: float) -> int:
    return round((time.monotonic() - start) * 1000)


def _refused(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ConnectionRefusedError)


def probe_http(url: str, expect_status: int, timeout: float) -> ProbeResult:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            code = response.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except (urllib.error.URLError, OSError) as exc:
        if _refused(exc):
            return ProbeResult(
                False, "down", f"Nothing is listening at {url} (connection refused)."
            )
        return ProbeResult(
            False, "unresponsive", f"{url} did not answer within {timeout:g}s: {exc}"
        )
    if code != expect_status:
        return ProbeResult(
            False,
            "unresponsive",
            f"{url} answered HTTP {code}, expected {expect_status}.",
            _timed(started),
        )
    return ProbeResult(True, "up", f"HTTP {code} from {url}", _timed(started))


def probe_wyoming(host: str, port: int, timeout: float) -> ProbeResult:
    """Wyoming servers answer a `describe` event with an `info` event."""
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(b'{"type": "describe"}\n')
            buffer = b""
            while b"\n" not in buffer:
                chunk = sock.recv(4096)
                if not chunk:
                    return ProbeResult(
                        False,
                        "unresponsive",
                        f"{host}:{port} accepted the connection but closed it without replying.",
                    )
                buffer += chunk
    except ConnectionRefusedError:
        return ProbeResult(
            False, "down", f"Nothing is listening on {host}:{port} (connection refused)."
        )
    except OSError as exc:
        return ProbeResult(
            False,
            "unresponsive",
            f"{host}:{port} did not answer a Wyoming describe in {timeout:g}s: {exc}",
        )
    try:
        header = json.loads(buffer.split(b"\n", 1)[0])
    except ValueError:
        return ProbeResult(
            False,
            "unresponsive",
            f"{host}:{port} replied, but not with Wyoming JSON: {buffer[:80]!r}",
        )
    if header.get("type") != "info":
        return ProbeResult(
            False,
            "unresponsive",
            f"{host}:{port} replied with {header.get('type')!r}, expected 'info'.",
        )
    return ProbeResult(True, "up", f"Wyoming info from {host}:{port}", _timed(started))


def probe_tcp(host: str, port: int, timeout: float) -> ProbeResult:
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except ConnectionRefusedError:
        return ProbeResult(
            False, "down", f"Nothing is listening on {host}:{port} (connection refused)."
        )
    except OSError as exc:
        return ProbeResult(
            False, "unresponsive", f"{host}:{port} did not accept a connection: {exc}"
        )
    return ProbeResult(True, "up", f"{host}:{port} accepts connections", _timed(started))


def run_probe(probe: Probe) -> ProbeResult:
    if probe.kind == "http":
        return probe_http(probe.url or "", probe.expect_status, probe.timeout)
    if probe.kind == "wyoming":
        return probe_wyoming(probe.host, probe.port or 0, probe.timeout)
    return probe_tcp(probe.host, probe.port or 0, probe.timeout)
