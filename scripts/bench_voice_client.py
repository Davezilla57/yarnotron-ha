"""End-to-end client latency: transcript in, first audible sample out.

This is the number phase 4's Rust client has to match or beat, and it did not
exist before: phases 1 and 2 were measured on the backend and on the
synthesiser separately, never through the real voice loop with a real speaker.

Drives LocalVoiceLoop with a supplied transcript, so it measures everything
after speech recognition -- backend stream, sentence gating, synthesis and
playback -- without needing someone to talk to it.

    uv run python scripts/bench_voice_client.py --api http://127.0.0.1:8123

Plays audio. Needs a foreground session with an audio device.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.voice.backend import BackendClient
from yarnotron.voice.loop import LocalVoiceLoop, VoiceLoopSettings
from yarnotron.voice.tts import DEFAULT_KOKORO_VOICE, create_speaker

TRANSCRIPTS = [
    "Wren hears something scratching at the door downstairs.",
    "It is a seal pup stuck in the netting.",
    "The storm gets worse and the windows start to rattle.",
]


class TimingSpeaker:
    """Wraps the real speaker and records when each utterance starts and ends."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.events: list[tuple[float, float]] = []

    def speak(self, text: str) -> None:
        started = time.perf_counter()
        self.inner.speak(text)
        self.events.append((started, time.perf_counter()))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8123")
    parser.add_argument("--story-id", default=None)
    parser.add_argument("--voice", default=DEFAULT_KOKORO_VOICE)
    parser.add_argument("--silent", action="store_true", help="skip playback, time synthesis only")
    args = parser.parse_args()

    backend = BackendClient(args.api)
    session = backend.create_session("bench_child", args.story_id, "bench")

    speaker = TimingSpeaker(
        create_speaker(provider="none" if args.silent else "auto", kokoro_voice=args.voice)
    )
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=None,  # transcripts are supplied, so the mic is never used
        transcriber=None,
        speaker=speaker,
        settings=VoiceLoopSettings(child_id="bench_child", story_id=args.story_id,
                                   device_id="bench"),
    )

    print(f"api={args.api}  voice={args.voice}  playback={not args.silent}\n")
    first_audio: list[float] = []
    totals: list[float] = []

    try:
        for transcript in TRANSCRIPTS:
            speaker.events.clear()
            started = time.perf_counter()
            turn = loop.run_turn(session_id=session["id"], transcript=transcript)
            total = (time.perf_counter() - started) * 1000
            if turn is None:
                print(f"  FAILED  {transcript[:44]}")
                continue

            # First audible sample is when the first utterance began playing.
            first = (speaker.events[0][0] - started) * 1000 if speaker.events else total
            first_audio.append(first)
            totals.append(total)
            spoken = sum(end - start for start, end in speaker.events) * 1000
            print(f"  first audio {first:7.0f} ms   reply spoken {spoken:7.0f} ms   "
                  f"turn {total:7.0f} ms   {len(speaker.events)} utterances")
            print(f"      {turn.response_text[:88]}")
    finally:
        backend.end_session(session["id"])

    if first_audio:
        print(f"\n{'metric':28s} {'p50':>9} {'p95':>9}")
        print("-" * 50)
        print(f"{'time to first audio (ms)':28s} {percentile(first_audio, 0.5):>9.0f} "
              f"{percentile(first_audio, 0.95):>9.0f}")
        print(f"{'whole turn (ms)':28s} {percentile(totals, 0.5):>9.0f} "
              f"{percentile(totals, 0.95):>9.0f}")
        print("\nThis is the phase 1-2 client baseline phase 4 must match or beat.")


if __name__ == "__main__":
    main()
