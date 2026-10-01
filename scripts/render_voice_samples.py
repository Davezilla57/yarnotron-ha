"""Render the same story replies in several voices, as WAV files to compare.

Phase 2 asks for a blind A/B across candidate voices. Playing audio from a
script needs an output device a background process cannot open, so write files
instead and listen to them however you like.

    uv run python scripts/render_voice_samples.py
    uv run python scripts/render_voice_samples.py --all          # all 54 voices
    uv run python scripts/render_voice_samples.py --voices af_heart bm_george
    uv run python scripts/render_voice_samples.py --text "Try this line."

Files land in bench/voice_samples/ named reply<N>__<voice>.wav, so the same
reply in different voices sorts together and the voice is obvious from the
name. Replies are real output from the golden corpus, not invented ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "bench" / "golden" / "responses.json"
OUT_DIR = REPO_ROOT / "bench" / "voice_samples"
MODEL_DIR = Path.home() / ".cache" / "kokoro"

# A spread across accent and gender rather than every voice, so the first
# listen is a handful of files instead of hundreds.
SHORTLIST = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "am_michael",
    "am_fenrir",
    "bf_emma",
    "bm_george",
]

FALLBACK_REPLIES = [
    "A scratching sound at the door adds such a mystery. What do you think is making it?",
    "A fisherman named Fennick is a wonderful addition. What does Fennick look like?",
    "A keeper looks after a place, like a lighthouse. How does Wren feel about her job?",
    "I love that she brings the lantern for extra light. What does she see?",
    "The lamp catches and the light sweeps over the water. Who sees it first?",
]


def corpus_replies(count: int) -> list[str]:
    """Real replies, preferring a spread of lengths over the first N."""
    if not GOLDEN.exists():
        return FALLBACK_REPLIES[:count]
    turns = json.loads(GOLDEN.read_text())["turns"]
    texts = [
        t["response"]["response_text"]
        for t in turns
        if t.get("response", {}).get("response_text")
        and t.get("actual_intent") in {"continue_story", "edit_story", "ask_definition"}
    ]
    if len(texts) < count:
        return (texts + FALLBACK_REPLIES)[:count]
    texts.sort(key=len)
    # Evenly spaced across the length range: a voice can sound fine on a short
    # line and tiring on a long one.
    step = len(texts) / count
    return [texts[min(int(i * step), len(texts) - 1)] for i in range(count)]


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> float:
    audio = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return len(audio) / sample_rate


def render_say(replies: list[str], voice: str) -> list[tuple[str, float, float]]:
    """The incumbent, for comparison. `say -o` writes a file and needs no device."""
    if shutil.which("say") is None:
        return []
    rows = []
    for index, reply in enumerate(replies, start=1):
        target = OUT_DIR / f"reply{index}__say-{voice}.wav"
        aiff = target.with_suffix(".aiff")
        started = time.perf_counter()
        subprocess.run(["say", "-v", voice, "-o", str(aiff), reply], check=True)
        elapsed = (time.perf_counter() - started) * 1000
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(aiff), str(target)],
            check=True,
            capture_output=True,
        )
        aiff.unlink(missing_ok=True)
        with wave.open(str(target)) as wav:
            duration = wav.getnframes() / wav.getframerate()
        rows.append((target.name, elapsed, duration))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", nargs="*", default=None, help="kokoro voice ids")
    parser.add_argument("--all", action="store_true", help="every kokoro voice")
    parser.add_argument("--replies", type=int, default=5)
    parser.add_argument("--text", default=None, help="one line instead of corpus replies")
    parser.add_argument("--say-voice", default="Samantha", help="incumbent to compare against")
    parser.add_argument("--no-say", action="store_true")
    args = parser.parse_args()

    from kokoro_onnx import Kokoro

    model, voices_file = MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin"
    if not model.exists() or not voices_file.exists():
        raise SystemExit(f"Kokoro model files not found in {MODEL_DIR}")

    kokoro = Kokoro(str(model), str(voices_file))
    available = sorted(kokoro.get_voices())
    if args.all:
        chosen = available
    elif args.voices:
        unknown = [v for v in args.voices if v not in available]
        if unknown:
            raise SystemExit(f"unknown voice(s): {', '.join(unknown)}")
        chosen = args.voices
    else:
        chosen = [v for v in SHORTLIST if v in available]

    replies = [args.text] if args.text else corpus_replies(args.replies)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"{len(replies)} replies x {len(chosen)} kokoro voices -> {OUT_DIR}\n")
    for index, reply in enumerate(replies, start=1):
        print(f"reply{index}: {reply}")
    print()

    rows: list[tuple[str, float, float]] = []
    for voice in chosen:
        for index, reply in enumerate(replies, start=1):
            started = time.perf_counter()
            samples, sample_rate = kokoro.create(reply, voice=voice, speed=1.0, lang="en-us")
            elapsed = (time.perf_counter() - started) * 1000
            target = OUT_DIR / f"reply{index}__{voice}.wav"
            duration = write_wav(target, samples, sample_rate)
            rows.append((target.name, elapsed, duration))

    if not args.no_say:
        rows.extend(render_say(replies, args.say_voice))

    print(f"{'file':44s} {'synth ms':>9} {'audio s':>8}")
    print("-" * 64)
    for name, elapsed, duration in rows:
        print(f"{name:44s} {elapsed:9.0f} {duration:8.2f}")

    print(f"\n{len(rows)} files in {OUT_DIR}")
    print(f"open {OUT_DIR}")


if __name__ == "__main__":
    main()
