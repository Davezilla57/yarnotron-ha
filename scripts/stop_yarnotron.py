#!/usr/bin/env python3
"""Stop local Yarnotron services."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".yarnotron"
BACKEND_PID = RUNTIME_DIR / "backend.pid"
VOICE_PID = RUNTIME_DIR / "voice.pid"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop Yarnotron locally.")
    parser.add_argument("--no-voice", action="store_true", help="Do not stop the voice loop.")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if not args.no_voice:
        stop_process("Voice loop", VOICE_PID, timeout=args.timeout)
    stop_process("FastAPI backend", BACKEND_PID, timeout=args.timeout)

    print("Yarnotron services stopped.", flush=True)
    return 0


def stop_process(label: str, pid_path: Path, *, timeout: float) -> None:
    pid = read_pid(pid_path)
    if pid is None:
        print(f"No {label} PID file found.", flush=True)
        return
    if not process_is_running(pid):
        print(f"{label} PID {pid} is not running.", flush=True)
        pid_path.unlink(missing_ok=True)
        return

    print(f"Stopping {label} PID {pid}...", flush=True)
    terminate_process_tree(pid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_running(pid):
            pid_path.unlink(missing_ok=True)
            print(f"{label} stopped.", flush=True)
            return
        time.sleep(0.25)

    print(f"{label} did not stop in time; forcing shutdown.", flush=True)
    kill_process_tree(pid)
    pid_path.unlink(missing_ok=True)


def terminate_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T"], check=False)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
