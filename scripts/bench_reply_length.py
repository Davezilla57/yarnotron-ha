"""Measure how long the corpus replies actually take to speak.

A reply spoken aloud is time the child spends listening instead of telling the
story, so its length is a product metric, not a cosmetic one.

Measure it, do not estimate it. A chars-per-second ratio taken from one sample
read 20% short (0.0463 against a real 0.0559), which was enough to turn a miss
into an apparent pass.

    uv run python scripts/bench_reply_length.py bench/golden/responses.json
"""
import json
import statistics
import sys
from pathlib import Path

from kokoro_onnx import Kokoro

d = Path.home() / ".cache" / "kokoro"
kokoro = Kokoro(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))

replies = [
    t["response"]["response_text"]
    for t in json.load(open(sys.argv[1]))["turns"]
    if t.get("response", {}).get("response_text")
]

durations, ratios = [], []
for reply in replies:
    samples, rate = kokoro.create(reply, voice="af_heart", speed=1.0, lang="en-us")
    seconds = len(samples) / rate
    durations.append(seconds)
    ratios.append(seconds / len(reply))

durations.sort()
mean = statistics.mean(durations)
print(f"measured over {len(durations)} replies, synthesised with kokoro af_heart\n")
print(f"  mean   {mean:5.2f}s")
print(f"  p50    {durations[len(durations) // 2]:5.2f}s")
print(f"  p95    {durations[int(0.95 * len(durations))]:5.2f}s")
print(f"  max    {durations[-1]:5.2f}s")
print(f"  over 5s: {sum(1 for x in durations if x > 5)}/{len(durations)}")
print(f"\n  seconds per character: {statistics.mean(ratios):.4f}")

