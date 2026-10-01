"""Measure two model-level costs that dominate the turn budget.

1. Thinking. Ollama thinks by default for models that support it. The turn path
   passes `think: false`, but this proves the flag is honoured and prices what
   it saves.
2. GPU co-residency. Whether keeping the safety model loaded alongside the
   story model costs anything, which decides whether the phase 1 plan to pin
   both models is worth doing. Measured at 0.99x — it is free — but generation
   throughput has been seen to vary ~3x between bursts for reasons not yet
   understood, so re-measure rather than trusting the recorded number.

    uv run python scripts/bench_models.py

This leaves both models resident on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.agent.prompts import build_story_turn_prompt
from yarnotron.core.config import get_settings
from yarnotron.models.agent import AgentTurnRequest, TurnIntent
from yarnotron.models.story import Character, Scene, Story, StoryFact
from yarnotron.services.ollama import OllamaClient

STORY = Story(
    title="The Lighthouse at Sable Point",
    synopsis="Wren and a stubborn mechanical gull keep the last lighthouse burning.",
    characters=[
        Character(name="Wren", role="keeper", traits=["brave"]),
        Character(name="Barnacle", role="companion", traits=["mechanical"]),
    ],
    scenes=[Scene(title="The Light Goes Out", summary="The lamp fails.", sequence=0)],
    facts=[StoryFact(text="The lamp runs on whale oil.")],
)


async def one_call(client: OllamaClient, model: str, prompt: str, think: bool | None) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 512},
    }
    if think is not None:
        body["think"] = think
    started = time.perf_counter()
    payload = await client._request("POST", "/api/generate", json=body)
    return {
        "wall_ms": (time.perf_counter() - started) * 1000,
        "eval": payload.get("eval_count") or 0,
        "eval_s": (payload.get("eval_duration") or 1) / 1e9,
        "thinking_chars": len(payload.get("thinking") or ""),
    }


def unload(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--story-model", default="gemma4:12b-mlx")
    parser.add_argument("--safety-model", default="gemma4:e2b-mlx")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    client = OllamaClient(get_settings())
    prompt = build_story_turn_prompt(
        request=AgentTurnRequest(transcript="Wren hears something scratching at the door."),
        intent=TurnIntent.CONTINUE_STORY,
        story=STORY,
        relevant_memories=[],
    )

    print(f"prompt: {len(prompt)} chars\n")
    print("=== thinking ===")
    print(f"{'model':20s} {'think':14s} {'wall ms':>9} {'eval tok':>9} {'tok/s':>7} {'thinking':>9}")
    print("-" * 72)
    for model in (args.story_model, args.safety_model):
        for think, label in ((False, "false"), (None, "omitted"), (True, "true")):
            result = await one_call(client, model, prompt, think)
            print(
                f"{model:20s} {label:14s} {result['wall_ms']:>9.0f} {result['eval']:>9d} "
                f"{result['eval'] / result['eval_s']:>7.1f} {result['thinking_chars']:>9d}"
            )

    # A/B/A/B rather than a single A then B: a one-shot comparison here
    # reported a 3x co-residency penalty that was really just the story model
    # still warming up in the A slot. Alternating and warming each phase
    # separates a real effect from an ordering artefact.
    print("\n=== gpu co-residency (think=false) ===")
    print(f"{'phase':26s} {'resident':34s} {'median tok/s':>13}")
    print("-" * 76)

    alone_rates: list[float] = []
    both_rates: list[float] = []
    for round_index in range(2):
        for co_resident, bucket in ((False, alone_rates), (True, both_rates)):
            if co_resident:
                await one_call(client, args.safety_model, prompt, False)
            else:
                unload(args.safety_model)
            await one_call(client, args.story_model, prompt, False)  # warm this phase
            rates = []
            for _ in range(args.runs):
                result = await one_call(client, args.story_model, prompt, False)
                rates.append(result["eval"] / result["eval_s"])
            rates.sort()
            median = rates[len(rates) // 2]
            bucket.append(median)
            label = f"{'A' if not co_resident else 'B'}{round_index + 1} story"
            label += " + safety" if co_resident else " alone"
            print(f"{label:26s} {','.join(_resident()):34s} {median:>13.1f}")

    rate_alone = sum(alone_rates) / len(alone_rates)
    rate_both = sum(both_rates) / len(both_rates)
    print(f"\nalone       : {rate_alone:.1f} tok/s")
    print(f"co-resident : {rate_both:.1f} tok/s")
    print(f"ratio       : {rate_alone / rate_both:.2f}x  (1.0 = co-residency is free)")

    await client.close()


def _resident() -> list[str]:
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    return [line.split()[0] for line in out.splitlines()[1:] if line.strip()]


if __name__ == "__main__":
    asyncio.run(main())
