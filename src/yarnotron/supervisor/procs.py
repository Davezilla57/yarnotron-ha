"""Start, find and stop operating-system processes."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path


def listening_pids(port: int) -> list[int]:
    """PIDs listening on a TCP port, via lsof. Empty when none, or lsof is missing."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [int(line) for line in result.stdout.split() if line.isdigit()]


def process_name(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() or "unknown"


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie that nobody has reaped still "exists"; treat it as gone.
    try:
        state = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    return not state.startswith("Z")


def spawn(
    command: tuple[str, ...],
    *,
    cwd: str | None,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen:
    """Start a detached process whose output goes to `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    log.write(
        f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(command)}\n".encode()
    )
    log.flush()
    try:
        return subprocess.Popen(
            list(command),
            cwd=os.path.expanduser(cwd) if cwd else None,
            env={**os.environ, **env},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives the supervisor restarting
        )
    finally:
        log.close()


def stop(pid: int, timeout: float) -> str:
    """Ask the process to exit (SIGTERM); escalate to SIGKILL after `timeout`.

    Returns "graceful", "killed" or "gone".
    """
    if not is_alive(pid):
        return "gone"
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return "graceful"
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    time.sleep(0.2)
    return "killed"


def log_tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])
