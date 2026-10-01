#!/usr/bin/env python3
"""Start the local Yarnotron service stack."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".yarnotron"
LOG_DIR = RUNTIME_DIR / "logs"
BACKEND_PID = RUNTIME_DIR / "backend.pid"
VOICE_PID = RUNTIME_DIR / "voice.pid"
BACKEND_LOG = LOG_DIR / "backend.log"
VOICE_LOG = LOG_DIR / "voice.log"
DEFAULT_WELCOME_MESSAGE = "Welcome to Yarnotron, what story would you like to work on?"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def exposure_warning(host: str, *, env: dict | None = None, dotenv: Path | None = None) -> str | None:
    """A warning when the API will be reachable from the network with no token.

    Binding to 0.0.0.0 is how a Home Assistant box on the LAN reaches Yarnotron.
    It also means every device on that LAN -- the child's included -- can open
    the dashboard, read stories and release safety blocks, unless a token is set.
    """
    if host in LOOPBACK_HOSTS:
        return None
    env = os.environ if env is None else env
    if (env.get("YARNOTRON_API_TOKEN") or "").strip():
        return None
    dotenv = ROOT / ".env" if dotenv is None else dotenv
    try:
        lines = dotenv.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "YARNOTRON_API_TOKEN" and value.strip().strip("\"'").strip():
            return None
    return (
        f"WARNING: binding to {host} with no YARNOTRON_API_TOKEN set. Anything on your "
        "network can read the stories and use the parent controls. Set a token in .env "
        "(see .env.example) before exposing Yarnotron beyond this machine."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Yarnotron locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8989)
    parser.add_argument("--no-backend", action="store_true", help="Do not start the FastAPI backend.")
    parser.add_argument("--no-reload", action="store_true", help="Run uvicorn without reload mode.")
    parser.add_argument("--sync", action="store_true", help="Run uv sync --extra dev --extra voice first.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Seconds to wait for /health.")
    parser.add_argument("--voice", action="store_true", help="Start the local microphone voice loop.")
    parser.add_argument("--story-id", help="Story id to pass to the voice loop.")
    parser.add_argument("--child-id", default="local_child", help="Child id to pass to the voice loop.")
    parser.add_argument("--device-id", default="startup-local-voice", help="Device id for voice sessions.")
    parser.add_argument("--input-device", help="Input device index or name for the voice loop.")
    parser.add_argument("--output-device", help="Output device index or name for Piper playback.")
    parser.add_argument(
        "--tts-provider",
        choices=["auto", "none", "macos-say", "piper"],
        default="auto",
        help="TTS provider for the voice loop.",
    )
    parser.add_argument("--tts-voice", help="Provider-specific voice name for the voice loop.")
    parser.add_argument(
        "--kokoro-voice",
        help="Kokoro voice id for the child-facing voice, such as af_heart or bm_george.",
    )
    parser.add_argument("--piper-model", help="Path to a Piper .onnx voice model.")
    parser.add_argument("--stt-model", default="base.en", help="faster-whisper model for the voice loop.")
    parser.add_argument("--stt-device", default="cpu", help="faster-whisper device for the voice loop.")
    parser.add_argument("--stt-compute-type", default="int8", help="faster-whisper compute type.")
    parser.add_argument("--wake-word", default="interpreter", help="Wake word for the voice loop.")
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="Never stop speaking to listen. Use if replies are being cut short.",
    )
    parser.add_argument(
        "--barge-in-margin",
        type=float,
        help="How much louder than its own echo a voice must be to interrupt (default 1.4).",
    )
    parser.add_argument(
        "--barge-in-frames",
        type=int,
        help="Speech frames out of 12 needed to interrupt (default 8).",
    )
    parser.add_argument(
        "--wake-response",
        default="I'm listening. What's your story?",
        help="Response spoken after the wake word is heard.",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=10.0,
        help="Seconds of silence before the voice loop returns to wake-word mode.",
    )
    parser.add_argument(
        "--no-welcome",
        action="store_true",
        help="Do not speak the startup welcome message.",
    )
    parser.add_argument(
        "--welcome-message",
        default=DEFAULT_WELCOME_MESSAGE,
        help="Startup message to speak after the backend health check passes.",
    )
    parser.add_argument(
        "--welcome-voice",
        help="Optional macOS say voice name for the startup welcome.",
    )
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    if args.sync:
        run(["uv", "sync", "--extra", "dev", "--extra", "voice"])

    if not args.no_backend:
        warning = exposure_warning(args.host)
        if warning:
            print(warning, file=sys.stderr, flush=True)
        start_backend(args.host, args.port, reload=not args.no_reload)
        api_url = f"http://{args.host}:{args.port}"
        wait_for_health(f"{api_url}/health", timeout=args.timeout)
        if args.voice:
            args.story_id = validate_voice_story_id(api_url, args.story_id)
        if not args.no_welcome:
            speak_welcome(args.welcome_message, voice=args.welcome_voice)
        if args.voice:
            start_voice_loop(args, api_url)

    print(flush=True)
    print("Yarnotron is ready.", flush=True)
    print(f"API:       http://{args.host}:{args.port}", flush=True)
    print(f"Dashboard: http://{args.host}:{args.port}/dashboard", flush=True)
    print(f"Backend log: {BACKEND_LOG}", flush=True)
    if args.voice:
        print(f"Voice log:   {VOICE_LOG}", flush=True)
    print(flush=True)
    print("Stop services with:", flush=True)
    print("  python3 scripts/stop_yarnotron.py", flush=True)
    return 0


def start_backend(host: str, port: int, *, reload: bool) -> None:
    existing_pid = read_pid(BACKEND_PID)
    if existing_pid is not None and process_is_running(existing_pid):
        print(f"FastAPI backend already appears to be running with PID {existing_pid}.", flush=True)
        return

    command = [
        "uv",
        "run",
        "uvicorn",
        "yarnotron.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

    print("Starting FastAPI backend...", flush=True)
    log_file = BACKEND_LOG.open("a", encoding="utf-8")
    log_file.write("\n=== Starting Yarnotron backend ===\n")
    log_file.flush()
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    BACKEND_PID.write_text(str(process.pid), encoding="utf-8")
    print(f"FastAPI backend PID: {process.pid}", flush=True)


def start_voice_loop(args: argparse.Namespace, api_url: str) -> None:
    existing_pid = read_pid(VOICE_PID)
    if existing_pid is not None and process_is_running(existing_pid):
        print(f"Voice loop already appears to be running with PID {existing_pid}.", flush=True)
        return

    command = [
        "uv",
        "run",
        "--extra",
        "voice",
        "scripts/local_voice_loop.py",
        "--api",
        api_url,
        "--child-id",
        args.child_id,
        "--device-id",
        args.device_id,
        "--tts-provider",
        args.tts_provider,
        "--stt-model",
        args.stt_model,
        "--stt-device",
        args.stt_device,
        "--stt-compute-type",
        args.stt_compute_type,
        "--wake-word",
        args.wake_word,
        "--wake-response",
        args.wake_response,
        "--idle-timeout-seconds",
        str(args.idle_timeout_seconds),
    ]
    optional_args = {
        "--story-id": args.story_id,
        "--input-device": args.input_device,
        "--output-device": args.output_device,
        "--tts-voice": args.tts_voice,
        "--kokoro-voice": args.kokoro_voice,
        "--piper-model": args.piper_model,
        "--barge-in-margin": args.barge_in_margin,
        "--barge-in-frames": args.barge_in_frames,
    }
    for flag, value in optional_args.items():
        if value:
            command.extend([flag, str(value)])
    if args.no_barge_in:
        command.append("--no-barge-in")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

    print("Starting microphone voice loop...", flush=True)
    log_file = VOICE_LOG.open("a", encoding="utf-8")
    log_file.write("\n=== Starting Yarnotron voice loop ===\n")
    log_file.flush()
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    VOICE_PID.write_text(str(process.pid), encoding="utf-8")
    print(f"Voice loop PID: {process.pid}", flush=True)


def validate_voice_story_id(api_url: str, story_id: str | None) -> str | None:
    if story_id is None:
        print(
            "Warning: no --story-id was provided. Voice turns will be saved, "
            "but they will not update a canonical story document.",
            flush=True,
        )
        return None

    if "..." in story_id:
        raise SystemExit(
            "--story-id story_... is a placeholder. Create or choose a story in "
            "the dashboard, then restart with its readable story id or exact title."
        )

    encoded_story_id = urllib.parse.quote(story_id, safe="")
    url = f"{api_url}/v1/stories/{encoded_story_id}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                return story_id
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise SystemExit(f"Could not validate story {story_id!r}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not validate story {story_id!r}: {exc}") from exc

    matched_story_id = find_story_id_by_title(api_url, story_ref=story_id)
    if matched_story_id is not None:
        print(
            f"Resolved story title {story_id!r} to story id {matched_story_id!r}.",
            flush=True,
        )
        return matched_story_id

    raise SystemExit(
        f"Story {story_id!r} was not found by id or exact title. Create or choose "
        "a story in the dashboard, then restart with its readable story id or title."
    )


def find_story_id_by_title(api_url: str, *, story_ref: str) -> str | None:
    url = f"{api_url}/v1/stories"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            stories = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not list stories while resolving {story_ref!r}: {exc}") from exc

    normalized_ref = " ".join(story_ref.casefold().split())
    matches = [
        story
        for story in stories
        if " ".join(str(story.get("title", "")).casefold().split()) == normalized_ref
    ]
    if len(matches) == 1:
        return str(matches[0]["id"])
    if len(matches) > 1:
        ids = ", ".join(str(story["id"]) for story in matches)
        raise SystemExit(
            f"Story title {story_ref!r} matches multiple stories. Restart with one "
            f"of these story ids: {ids}"
        )
    return None


def wait_for_health(url: str, *, timeout: float) -> None:
    print(f"Waiting for backend health at {url}...", flush=True)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print("Backend health check passed.", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.5)

    raise SystemExit(f"Backend did not become healthy within {timeout:.0f}s: {last_error}")


def speak_welcome(message: str, *, voice: str | None = None) -> None:
    if not message.strip():
        return
    print(f'Speaking welcome: "{message}"', flush=True)
    if shutil.which("say") is not None:
        command = ["say"]
        if voice:
            command.extend(["-v", voice])
        command.append(message)
        subprocess.run(command, check=False)
        return
    if os.name == "nt" and shutil.which("powershell") is not None:
        escaped_message = message.replace("'", "''")
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Add-Type -AssemblyName System.Speech; "
                    "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$speaker.Speak('{escaped_message}')"
                ),
            ],
            check=False,
        )
        return
    if shutil.which("spd-say") is not None:
        subprocess.run(["spd-say", message], check=False)
        return
    print("No local speech command found; welcome announcement skipped.", flush=True)


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
