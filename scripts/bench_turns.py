"""Replay the fixed corpus against a fresh story and report per-stage latency.

This is the phase 0 baseline harness. Every later phase is judged against the
stage table this prints, so the corpus and the seed story must not drift.

    uv run python scripts/bench_turns.py --repeat 3
    uv run python scripts/bench_turns.py --write-golden

Requires Ollama to be running. The datastore is a SQLite file, created on
first use.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.agent.safety import ModelSafetyReviewer
from yarnotron.agent.turn_processor import TurnProcessor
from yarnotron.core.config import get_settings
from yarnotron.models.agent import AgentTurnRequest
from yarnotron.models.operations import CreateSessionRequest
from yarnotron.models.story import CreateStoryRequest, StoryPatch, StoryPatchOperation
from yarnotron.services.memory import OllamaEmbedder, StoryMemoryBuilder
from yarnotron.services.ollama import OllamaClient
from yarnotron.services.repository import open_story_repository
from yarnotron.services.story_store import StoryRepository

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "bench" / "corpus.json"
GOLDEN_DIR = REPO_ROOT / "bench" / "golden"

TIMING_PREFIX = "timing "
# Most ids are uuid4 hex, but story ids are readable slugs that gain an
# incrementing suffix each time the same title is created (..._2, ..._3), so
# they have to be normalised too or the golden file churns on every rerun.
HEX_ID = re.compile(
    r"^(story|turn|character|scene|fact|chunk|session|media|event|version)_[0-9a-f]{32}$"
)
SLUG_STORY_ID = re.compile(r"^story_[a-z0-9]+(?:_[a-z0-9]+)*$")


def _id_kind(value: str) -> str | None:
    if HEX_ID.match(value):
        return value.split("_")[0]
    if SLUG_STORY_ID.match(value):
        return "story"
    return None


async def seed_story(repository: StoryRepository, seed: dict[str, Any]) -> str:
    """Create the fixed story, then build it out with real patch operations."""
    story = await repository.create(
        CreateStoryRequest(title=seed["title"], synopsis=seed["synopsis"])
    )
    for character in seed["characters"]:
        await repository.apply_patch(
            story.id,
            StoryPatch(operation=StoryPatchOperation.ADD_CHARACTER, data=character),
        )
    for scene in seed["scenes"]:
        await repository.apply_patch(
            story.id,
            StoryPatch(operation=StoryPatchOperation.ADD_SCENE, data=scene),
        )
    for fact in seed["facts"]:
        await repository.apply_patch(
            story.id,
            StoryPatch(operation=StoryPatchOperation.ADD_FACT, data=fact),
        )
    return story.id


def normalize(value: Any, ids: dict[str, str] | None = None) -> Any:
    """Strip values that change on every run so goldens diff meaningfully.

    Generated ids and timestamps carry no behavioural information but differ on
    every replay. Ids are replaced with stable per-response ordinals rather than
    a single placeholder, so a patch that targets the second character still
    reads as targeting the second character. Substance -- response text,
    intents, patch operations and payloads, safety decisions -- is exact.
    """
    if ids is None:
        ids = {}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"created_at", "updated_at", "ended_at", "started_at"}:
                result[key] = "<timestamp>"
            elif key in {"memory_chunk_ids", "retrieved_memory_ids", "media_asset_ids"}:
                result[key] = f"<{len(item)} ids>" if isinstance(item, list) else "<ids>"
            else:
                result[key] = normalize(item, ids)
        return result
    if isinstance(value, list):
        return [normalize(item, ids) for item in value]
    if isinstance(value, str):
        kind = _id_kind(value)
        if kind is not None:
            if value not in ids:
                seen = sum(1 for placeholder in ids.values() if placeholder.startswith(f"<{kind}_"))
                ids[value] = f"<{kind}_{seen + 1}>"
            return ids[value]
        with contextlib.suppress(ValueError):
            datetime.fromisoformat(value)
            return "<timestamp>"
    return value


async def run_pass(
    processor: TurnProcessor,
    repository: StoryRepository,
    corpus: dict[str, Any],
    limit: int | None,
    stream: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay every turn once against a freshly seeded story."""
    story_id = await seed_story(repository, corpus["seed_story"])
    session = await repository.create_session(
        CreateSessionRequest(child_id="bench_child", story_id=story_id, device_id="bench")
    )
    turns = corpus["turns"][:limit] if limit else corpus["turns"]

    timings: list[dict[str, Any]] = []
    goldens: list[dict[str, Any]] = []

    for entry in turns:
        request = AgentTurnRequest(
            child_id="bench_child",
            story_id=story_id,
            session_id=session.id,
            device_id="bench",
            transcript=entry["transcript"],
        )
        captured = io.StringIO()
        started = time.perf_counter()
        first_audio: list[float] = []

        async def on_sentence(sentence: str) -> None:
            if not first_audio:
                first_audio.append((time.perf_counter() - started) * 1000)

        try:
            with contextlib.redirect_stdout(captured):
                response = await processor.process(
                    request, on_sentence=on_sentence if stream else None
                )
        except Exception as exc:  # a bench must survive a bad turn and report it
            elapsed = (time.perf_counter() - started) * 1000
            goldens.append(
                {
                    "id": entry["id"],
                    "transcript": entry["transcript"],
                    "expect_intent": entry["expect_intent"],
                    "actual_intent": "<error>",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"  {entry['id']}  {'<error>':18s}  {elapsed:8.1f} ms  "
                f"{type(exc).__name__}: {str(exc)[:60]}",
                flush=True,
            )
            continue

        timing = _parse_timing(captured.getvalue())
        timing["turn"] = entry["id"]
        if first_audio:
            timing["first_audio_ms"] = round(first_audio[0], 1)
        timings.append(timing)

        goldens.append(
            {
                "id": entry["id"],
                "transcript": entry["transcript"],
                "expect_intent": entry["expect_intent"],
                "actual_intent": response.intent.value,
                "response": normalize(response.model_dump(mode="json")),
            }
        )
        print(
            f"  {entry['id']}  {response.intent.value:18s}"
            f"  {timing.get('total_ms', 0):8.1f} ms  {entry['transcript'][:52]}",
            flush=True,
        )

    await repository.delete(story_id)
    return timings, goldens


def _parse_timing(captured: str) -> dict[str, Any]:
    for line in captured.splitlines():
        if line.startswith(TIMING_PREFIX):
            return json.loads(line[len(TIMING_PREFIX) :])
    return {}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def print_stage_table(timings: list[dict[str, Any]]) -> None:
    stages: dict[str, list[float]] = {}
    for timing in timings:
        for key, value in timing.items():
            if key.endswith("_ms") and isinstance(value, (int, float)):
                stages.setdefault(key[:-3], []).append(float(value))

    ordered = sorted(stages.items(), key=lambda item: -statistics.median(item[1]))
    width = max(len(name) for name, _ in ordered)

    print()
    print(f"{'stage'.ljust(width)}  {'n':>4}  {'p50 ms':>9}  {'p95 ms':>9}  {'max ms':>9}")
    print("-" * (width + 38))
    for name, values in ordered:
        marker = "  <-- total" if name == "total" else ""
        print(
            f"{name.ljust(width)}  {len(values):>4}  "
            f"{percentile(values, 0.50):>9.1f}  "
            f"{percentile(values, 0.95):>9.1f}  "
            f"{max(values):>9.1f}{marker}"
        )


def print_intent_table(timings: list[dict[str, Any]]) -> None:
    by_intent: dict[str, list[float]] = {}
    for timing in timings:
        intent = str(timing.get("intent", "unknown"))
        if "total_ms" in timing:
            by_intent.setdefault(intent, []).append(float(timing["total_ms"]))

    ordered = sorted(by_intent.items(), key=lambda item: -statistics.median(item[1]))
    width = max(len(name) for name, _ in ordered)

    print()
    print(f"{'intent'.ljust(width)}  {'n':>4}  {'p50 ms':>9}  {'p95 ms':>9}")
    print("-" * (width + 28))
    for name, values in ordered:
        print(
            f"{name.ljust(width)}  {len(values):>4}  "
            f"{percentile(values, 0.50):>9.1f}  {percentile(values, 0.95):>9.1f}"
        )


def check_intents(goldens: list[dict[str, Any]]) -> None:
    completed = [g for g in goldens if g["actual_intent"] != "<error>"]
    mismatches = [g for g in completed if g["expect_intent"] != g["actual_intent"]]
    if not mismatches:
        print(f"\nintent routing: all {len(completed)} completed turns matched the corpus")
    else:
        print(f"\nintent routing: {len(mismatches)} turn(s) diverged from the corpus expectation")
        for entry in mismatches:
            print(f"  {entry['id']}  expected {entry['expect_intent']} -> got {entry['actual_intent']}")


def print_drift_table(timings: list[dict[str, Any]]) -> None:
    """Show whether a turn gets slower as the story grows underneath it.

    The prompt inlines the whole story document, so cost is expected to climb
    across a session. This is the number the phase 5 story compiler has to fix.
    """
    ordered = [t for t in timings if "total_ms" in t]
    if len(ordered) < 6:
        return
    third = len(ordered) // 3
    buckets = [
        ("first third", ordered[:third]),
        ("middle third", ordered[third : 2 * third]),
        ("last third", ordered[2 * third :]),
    ]

    print()
    print(f"{'session position':18s} {'n':>4} {'total p50':>11} {'generate p50':>14}")
    print("-" * 50)
    for label, bucket in buckets:
        totals = [float(t["total_ms"]) for t in bucket]
        generates = [float(t["model_generate_ms"]) for t in bucket if "model_generate_ms" in t]
        print(
            f"{label:18s} {len(bucket):>4} {percentile(totals, 0.5):>11.1f} "
            f"{percentile(generates, 0.5) if generates else 0:>14.1f}"
        )


def report_errors(goldens: list[dict[str, Any]]) -> None:
    failures = [g for g in goldens if g["actual_intent"] == "<error>"]
    if not failures:
        print(f"turn failures: none ({len(goldens)}/{len(goldens)} completed)")
        return
    counts: dict[str, int] = {}
    for entry in failures:
        counts[entry["error"].split(":")[0]] = counts.get(entry["error"].split(":")[0], 0) + 1
    print(f"turn failures: {len(failures)}/{len(goldens)} turns raised")
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {count:>3}x  {name}")
    print(f"  failed turns: {', '.join(e['id'] for e in failures)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1, help="replay passes (fresh story each)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N turns")
    parser.add_argument("--stream", action="store_true",
                        help="stream sentences and record time to first audio")
    parser.add_argument("--write-golden", action="store_true", help="record responses as the golden corpus")
    parser.add_argument("--golden-path", default=None, help="write the golden file somewhere else")
    parser.add_argument("--model", default=None, help="override the story model")
    parser.add_argument("--safety-model", default=None, help="override the safety model")
    parser.add_argument("--seed", type=int, default=None,
                        help="pin the sampling seed so the run is reproducible")
    args = parser.parse_args()

    corpus = json.loads(CORPUS_PATH.read_text())
    settings = get_settings()
    # Pin models explicitly so a baseline does not depend on whether a .env
    # happens to be present. settings.safety_model defaults to None, which
    # OllamaClient silently resolves to the (much larger) story model.
    overrides = {}
    if args.model:
        overrides["default_model"] = args.model
    if args.safety_model:
        overrides["safety_model"] = args.safety_model
    if args.seed is not None:
        overrides["model_seed"] = args.seed
    if overrides:
        settings = settings.model_copy(update=overrides)

    print(f"model={settings.default_model}  safety={settings.safety_model}  db={settings.database_url}")

    all_timings: list[dict[str, Any]] = []
    last_goldens: list[dict[str, Any]] = []
    async with open_story_repository(settings) as repository:
        ollama = OllamaClient(settings)
        processor = TurnProcessor(
            ollama=ollama,
            repository=repository,
            embedder=OllamaEmbedder(settings=settings, ollama=ollama),
            memory_builder=StoryMemoryBuilder(),
            safety=ModelSafetyReviewer(ollama=ollama, settings=settings),
            # A measurement pass must replay all 56 turns. The corpus is benign,
            # but the safety model is not deterministic about it -- a recorded
            # pass flagged 5 turns, exactly the production limit -- and a run
            # that ends itself at turn 40 is a broken measurement, not a
            # finding. Strike behaviour has its own tests.
            strike_limit=len(corpus["turns"]) + 1,
        )
        try:
            for index in range(args.repeat):
                print(f"\npass {index + 1}/{args.repeat}")
                timings, goldens = await run_pass(
                    processor, repository, corpus, args.limit, args.stream
                )
                all_timings.extend(timings)
                last_goldens = goldens
        finally:
            await ollama.close()

    print_stage_table(all_timings)
    print_intent_table(all_timings)
    print_drift_table(all_timings)
    check_intents(last_goldens)
    report_errors(last_goldens)

    if args.write_golden:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        target = Path(args.golden_path) if args.golden_path else GOLDEN_DIR / "responses.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "config": {
                        "default_model": settings.default_model,
                        "safety_model": settings.safety_model or f"(unset -> {settings.default_model})",
                        "embedding_model": settings.embedding_model,
                        "model_seed": settings.model_seed,
                    },
                    "turns": last_goldens,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"\nwrote {len(last_goldens)} golden responses to {target}")


if __name__ == "__main__":
    asyncio.run(main())
