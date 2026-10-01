"""Measure real faster-whisper transcription time on the corpus transcripts.

The architecture teardown lists STT as the one stage still estimated rather
than measured. This synthesises each corpus transcript to 16 kHz mono WAV with
the macOS speech synthesiser, then transcribes it with the same
FasterWhisperTranscriber the voice loop uses, so the number is real work on
real audio rather than a guess.

    uv sync --extra voice --extra dev
    uv run python scripts/bench_stt.py

Synthesised speech is cleaner than a child at a microphone, so treat the
result as a floor: real transcription will be at or above this.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.voice.stt import FasterWhisperTranscriber, TranscriptionSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "bench" / "corpus.json"
AUDIO_DIR = REPO_ROOT / "bench" / "audio"

# voice/audio.py records at this rate; matching it keeps the measurement honest.
SAMPLE_RATE = 16_000
# voice/audio.py waits this long after speech stops before transcription starts.
VAD_TAIL_SECONDS = 0.9


def synthesize(text: str, target: Path, voice: str) -> None:
    aiff = target.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(aiff), str(target)],
        check=True,
        capture_output=True,
    )
    aiff.unlink()


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="Samantha", help="macOS `say` voice")
    parser.add_argument("--model", default="base.en", help="faster-whisper model")
    parser.add_argument("--limit", type=int, default=None, help="only the first N transcripts")
    parser.add_argument("--keep-audio", action="store_true", help="leave the synthesised WAVs on disk")
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text())
    turns = corpus["turns"][: args.limit] if args.limit else corpus["turns"]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"synthesising {len(turns)} utterances with `say -v {args.voice}` at {SAMPLE_RATE} Hz")
    clips: list[tuple[str, Path, str]] = []
    for entry in turns:
        target = AUDIO_DIR / f"{entry['id']}.wav"
        if not target.exists():
            synthesize(entry["transcript"], target, args.voice)
        clips.append((entry["id"], target, entry["transcript"]))

    transcriber = FasterWhisperTranscriber(TranscriptionSettings(model=args.model))

    load_started = time.perf_counter()
    transcriber._load_model()
    load_ms = (time.perf_counter() - load_started) * 1000
    print(f"model load ({args.model}, cpu/int8): {load_ms:.0f} ms  [once per process]\n")

    durations: list[float] = []
    latencies: list[float] = []
    exact = 0

    for turn_id, path, expected in clips:
        audio_seconds = wav_seconds(path)
        started = time.perf_counter()
        text = transcriber.transcribe(path)
        elapsed_ms = (time.perf_counter() - started) * 1000

        durations.append(audio_seconds)
        latencies.append(elapsed_ms)
        match = _loose_match(text, expected)
        exact += match
        print(
            f"  {turn_id}  audio {audio_seconds:5.2f}s  stt {elapsed_ms:7.1f} ms  "
            f"rtf {elapsed_ms / 1000 / audio_seconds:5.2f}  {'ok ' if match else 'DIFF'}  {text[:44]}"
        )

    print()
    print(f"{'metric':28s} {'p50':>10} {'p95':>10} {'max':>10}")
    print("-" * 62)
    print(
        f"{'stt latency (ms)':28s} {percentile(latencies, 0.5):>10.1f} "
        f"{percentile(latencies, 0.95):>10.1f} {max(latencies):>10.1f}"
    )
    print(
        f"{'audio duration (s)':28s} {percentile(durations, 0.5):>10.2f} "
        f"{percentile(durations, 0.95):>10.2f} {max(durations):>10.2f}"
    )
    rtfs = [ms / 1000 / sec for ms, sec in zip(latencies, durations)]
    print(
        f"{'real-time factor':28s} {percentile(rtfs, 0.5):>10.2f} "
        f"{percentile(rtfs, 0.95):>10.2f} {max(rtfs):>10.2f}"
    )
    print()
    print(f"transcripts matching input: {exact}/{len(clips)}")
    print(
        f"\nend of speech -> transcript p50: {VAD_TAIL_SECONDS * 1000 + percentile(latencies, 0.5):.0f} ms"
        f"  ({VAD_TAIL_SECONDS * 1000:.0f} ms VAD tail + {percentile(latencies, 0.5):.0f} ms stt)"
    )

    if not args.keep_audio:
        for _, path, _ in clips:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            AUDIO_DIR.rmdir()


def _loose_match(actual: str, expected: str) -> bool:
    strip = str.maketrans("", "", ".,!?'\"")
    return actual.lower().translate(strip).split() == expected.lower().translate(strip).split()


if __name__ == "__main__":
    main()
