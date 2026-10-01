"""Check what Yarnotron needs, and fetch what can be fetched.

Everything else is one command; this is the piece that was still a manual
download. Run it once after cloning, and any time something stops working and
you want to know which part is missing.

    uv run python scripts/setup_yarnotron.py           # report only
    uv run python scripts/setup_yarnotron.py --install # fetch what is missing

Checks Ollama and its models, the Kokoro voice files, the speech-to-text model,
and the data directory. Reports each as ok or missing with the command that
fixes it, so a failure names its own remedy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.core.config import application_support_dir, get_settings
from yarnotron.voice.tts import KOKORO_MODEL_NAME, KOKORO_VOICES_NAME, kokoro_model_paths

KOKORO_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
OK, MISSING, WARN = "ok", "MISSING", "warn"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append((name, status, detail))

    def render(self) -> bool:
        width = max(len(name) for name, _, _ in self.rows)
        print(f"\n{'component'.ljust(width)}  {'status':<8} detail")
        print("-" * (width + 60))
        for name, status, detail in self.rows:
            print(f"{name.ljust(width)}  {status:<8} {detail}")
        missing = [name for name, status, _ in self.rows if status == MISSING]
        print()
        if missing:
            print(f"missing: {', '.join(missing)}")
            print("run again with --install to fetch what can be fetched")
        else:
            print("everything Yarnotron needs is present")
            print("\nstart it with:\n  python3 scripts/start_yarnotron.py --voice")
        return not missing


def ollama_models() -> list[str] | None:
    try:
        with urllib.request.urlopen(
            f"{get_settings().ollama_base_url.rstrip('/')}/api/tags", timeout=5
        ) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return [model.get("name", "") for model in payload.get("models", [])]


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {target.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.rename(target)
    print(f"  {target.name} ({target.stat().st_size / 1e6:.0f} MB)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="fetch what is missing")
    args = parser.parse_args()

    settings = get_settings()
    report = Report()

    # --- data directory -------------------------------------------------
    data_dir = application_support_dir()
    report.add(
        "data directory",
        OK,
        f"{data_dir}" + ("" if data_dir.exists() else "  (created on first run)"),
    )

    # --- ollama ---------------------------------------------------------
    installed = ollama_models()
    if installed is None:
        report.add("ollama", MISSING, f"not reachable at {settings.ollama_base_url} -- run: ollama serve")
    else:
        report.add("ollama", OK, f"{len(installed)} models at {settings.ollama_base_url}")
        if settings.safety_model is None:
            # Phase 0 finding: an unset safety model silently resolves to the
            # story model, which works but costs about 2.5x per review.
            report.add(
                "safety model",
                WARN,
                "unset, falling back to the story model (~2.5x slower) -- "
                "set YARNOTRON_SAFETY_MODEL=gemma4:e2b-mlx",
            )
        for label, name in (
            ("story model", settings.default_model),
            ("safety model", settings.safety_model),
            ("embedding model", settings.embedding_model or ""),
        ):
            if not name:
                continue
            present = any(model == name or model.startswith(f"{name}:") for model in installed)
            report.add(
                label,
                OK if present else MISSING,
                name if present else f"{name} -- run: ollama pull {name}",
            )

    # --- kokoro ---------------------------------------------------------
    model, voices = kokoro_model_paths()
    for path, filename in ((model, KOKORO_MODEL_NAME), (voices, KOKORO_VOICES_NAME)):
        if path.exists():
            report.add(f"kokoro {filename}", OK, f"{path.stat().st_size / 1e6:.0f} MB")
            continue
        if args.install:
            try:
                download(f"{KOKORO_RELEASE}/{filename}", path)
                report.add(f"kokoro {filename}", OK, f"{path.stat().st_size / 1e6:.0f} MB")
                continue
            except Exception as exc:  # network, disk, anything
                report.add(f"kokoro {filename}", MISSING, f"download failed: {exc}")
                continue
        report.add(f"kokoro {filename}", MISSING, f"{path} -- run with --install")

    # --- speech to text -------------------------------------------------
    try:
        import faster_whisper  # noqa: F401

        report.add("speech to text", OK, "faster-whisper installed, model downloads on first use")
    except ImportError:
        report.add("speech to text", MISSING, "run: uv sync --extra voice")

    # --- audio ----------------------------------------------------------
    try:
        import sounddevice as sd

        report.add(
            "audio devices",
            OK,
            f"in: {sd.query_devices(kind='input')['name']} / "
            f"out: {sd.query_devices(kind='output')['name']}",
        )
    except Exception as exc:
        report.add("audio devices", WARN, f"unavailable: {type(exc).__name__} (backend still works)")

    return 0 if report.render() else 1


if __name__ == "__main__":
    raise SystemExit(main())
