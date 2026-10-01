"""Does Kokoro stream, and what does its first audio chunk cost?

Phase 1 got first audio to 2806 ms by speaking each sentence as the model
writes it. That win only survives phase 2 if the synthesiser also starts
producing audio quickly -- a synthesiser that emits whole utterances hands the
latency straight back.

The number that matters is **time to first audio chunk**, not total synthesis
time: the child hears the first chunk, and the rest streams while they listen.

    uv run python scripts/bench_tts.py
    uv run python scripts/bench_tts.py --voice af_heart --engine kokoro

Compares against macOS `say`, which is what the voice loop uses today.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path.home() / ".cache" / "kokoro"
SAMPLE_RATE = 24_000

# Representative replies: what the model actually produces, first sentence first.
SENTENCES = [
    "That's a chilling sound for a night this cold!",
    "I've added that mysterious scratching to our story.",
    "What do you think is making that noise outside the door?",
    "Oh, a seal pup! That is such a sweet and unexpected surprise.",
    "Wren climbed the two hundred steps, counting each one aloud.",
]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)]


def measure_say(sentences: list[str], voice: str) -> tuple[list[float], list[float]]:
    """macOS `say` writing a file -- no streaming, so first audio is the whole file."""
    first, total = [], []
    out = REPO_ROOT / "bench" / "_tts_tmp.aiff"
    out.parent.mkdir(parents=True, exist_ok=True)
    for sentence in sentences:
        started = time.perf_counter()
        subprocess.run(["say", "-v", voice, "-o", str(out), sentence], check=True)
        elapsed = (time.perf_counter() - started) * 1000
        first.append(elapsed)
        total.append(elapsed)
    out.unlink(missing_ok=True)
    return first, total


def measure_kokoro(sentences: list[str], voice: str) -> tuple[list[float], list[float]]:
    from kokoro_onnx import Kokoro

    model = MODEL_DIR / "kokoro-v1.0.onnx"
    voices = MODEL_DIR / "voices-v1.0.bin"
    if not model.exists() or not voices.exists():
        raise SystemExit(
            f"Kokoro model files not found in {MODEL_DIR}.\n"
            "Download kokoro-v1.0.onnx and voices-v1.0.bin from\n"
            "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
        )

    load_started = time.perf_counter()
    kokoro = Kokoro(str(model), str(voices))
    print(f"kokoro model load: {(time.perf_counter() - load_started) * 1000:.0f} ms  [once]")

    # Warm: the first synthesis pays one-off graph setup.
    kokoro.create("Warming up.", voice=voice, speed=1.0, lang="en-us")

    first, total = [], []
    for sentence in sentences:
        started = time.perf_counter()
        samples, rate = kokoro.create(sentence, voice=voice, speed=1.0, lang="en-us")
        elapsed = (time.perf_counter() - started) * 1000
        first.append(elapsed)
        total.append(elapsed)
        seconds = len(samples) / rate
        print(f"  {elapsed:7.0f} ms for {seconds:4.2f}s audio  rtf {elapsed / 1000 / seconds:4.2f}"
              f"  {sentence[:46]}")
    return first, total


def measure_kokoro_stream(sentences: list[str], voice: str) -> list[float]:
    """Time to the first chunk when using the streaming API, if there is one."""
    import asyncio

    from kokoro_onnx import Kokoro

    kokoro = Kokoro(str(MODEL_DIR / "kokoro-v1.0.onnx"), str(MODEL_DIR / "voices-v1.0.bin"))
    if not hasattr(kokoro, "create_stream"):
        return []

    async def run() -> list[float]:
        firsts = []
        await _drain(kokoro.create_stream("Warming up.", voice=voice, lang="en-us"))
        for sentence in sentences:
            started = time.perf_counter()
            stream = kokoro.create_stream(sentence, voice=voice, speed=1.0, lang="en-us")
            async for _samples, _rate in stream:
                firsts.append((time.perf_counter() - started) * 1000)
                break
            await _drain(stream)
        return firsts

    return asyncio.run(run())


async def _drain(stream) -> None:
    try:
        async for _ in stream:
            pass
    except Exception:
        pass


def report(label: str, first: list[float], total: list[float]) -> None:
    print(f"\n{label}")
    print(f"  time to first audio  p50 {percentile(first, 0.5):7.0f} ms   "
          f"p95 {percentile(first, 0.95):7.0f} ms")
    if total is not first:
        print(f"  total synthesis      p50 {percentile(total, 0.5):7.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["say", "kokoro", "both"], default="both")
    parser.add_argument("--voice", default="af_heart", help="kokoro voice")
    parser.add_argument("--say-voice", default="Samantha")
    args = parser.parse_args()

    print(f"{len(SENTENCES)} sentences, one synthesis call each\n")

    if args.engine in {"say", "both"}:
        first, total = measure_say(SENTENCES, args.say_voice)
        report(f"macOS say ({args.say_voice})  -- today's engine, no streaming", first, total)

    if args.engine in {"kokoro", "both"}:
        first, total = measure_kokoro(SENTENCES, args.voice)
        report(f"kokoro-onnx ({args.voice}) whole sentence", first, total)

        streamed = measure_kokoro_stream(SENTENCES, args.voice)
        if streamed:
            report(f"kokoro-onnx ({args.voice}) streaming, first chunk", streamed, streamed)
        else:
            print("\nkokoro-onnx exposes no create_stream: whole-sentence only.")
            print("Per-sentence synthesis is then the unit of streaming, which is")
            print("what the voice loop already does.")


if __name__ == "__main__":
    main()
