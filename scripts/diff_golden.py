"""Compare two golden corpus files and report what actually changed.

Used to answer "did this change alter behaviour?" -- for streaming in phase 1,
and eventually for the Rust port in phase 5, where the Python implementation is
the correctness oracle.

    uv run python scripts/diff_golden.py before.json after.json
    uv run python scripts/diff_golden.py before.json after.json --show 20

`response_text` is generated prose. gemma4 at temperature 0 is close to
deterministic but not exactly so, so text differences are expected and are not
by themselves a regression. The columns that must hold are intent, safety
action, and the shape of the story patches -- those are decisions, not wording.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def patch_shape(patches: list[dict[str, Any]] | None) -> list[tuple[str, tuple[str, ...]]]:
    """What each patch does, ignoring prose and unset optional fields.

    Keys whose value is null are dropped: the model including or omitting
    `source_turn_id` is not a different decision, and counting it as one hides
    the differences that matter.
    """
    shapes = []
    for patch in patches or []:
        data = patch.get("data") or {}
        keys = tuple(sorted(k for k, v in data.items() if v is not None))
        shapes.append((patch.get("operation", "?"), keys))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--show", type=int, default=8, help="how many differing turns to print")
    args = parser.parse_args()

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    a = {t["id"]: t for t in before["turns"]}
    b = {t["id"]: t for t in after["turns"]}

    print(f"A {args.before}: {before.get('config')}")
    print(f"B {args.after}: {after.get('config')}")

    shared = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        print(f"\nturns only in A: {only_a}\nturns only in B: {only_b}")
    print(f"\ncomparing {len(shared)} shared turns\n")

    counts = dict.fromkeys(("intent", "safety", "patch shape", "patch payload", "text"), 0)
    diffs = []
    for tid in shared:
        ra = a[tid].get("response", {})
        rb = b[tid].get("response", {})
        checks = {
            "intent": a[tid]["actual_intent"] == b[tid]["actual_intent"],
            "safety": ra.get("safety", {}).get("action") == rb.get("safety", {}).get("action"),
            "patch shape": patch_shape(ra.get("story_patches")) == patch_shape(rb.get("story_patches")),
            "patch payload": ra.get("story_patches") == rb.get("story_patches"),
            "text": ra.get("response_text") == rb.get("response_text"),
        }
        for key, ok in checks.items():
            counts[key] += ok
        if not all(checks.values()):
            diffs.append((tid, checks, ra, rb))

    # Decisions must match. Prose need not: `response_text` is generated
    # wording, and a patch payload carries generated wording too (fact text,
    # scene summaries). Even with a pinned seed those drift, because the prompt
    # itself differs run to run -- story ids increment and retrieved memory
    # changes -- so the model is not being asked the same question twice.
    DECISIONS = ("intent", "safety", "patch shape")
    total = len(shared)
    print(f"{'field':16s} {'identical':>12}   verdict")
    print("-" * 66)
    for key, value in counts.items():
        is_decision = key in DECISIONS
        expected = "MUST match" if is_decision else "prose, drift expected"
        flag = "ok" if value == total else ("REGRESSION" if is_decision else "drift")
        print(f"{key:16s} {value:>6}/{total:<5} {flag:>12}   {expected}")

    decisions = [k for k in DECISIONS if counts[k] != total]
    print()
    if decisions:
        print(f"DECISIONS CHANGED in: {', '.join(decisions)} -- investigate before shipping.")
    else:
        print("No decision changed: same intents, same safety actions, same patch operations.")

    print(f"\n{len(diffs)} turns differ in some field; showing {min(args.show, len(diffs))}")
    for tid, checks, ra, rb in diffs[: args.show]:
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        print(f"\n  {tid}  differs in: {failed}")
        if not checks["text"]:
            print(f"    A: {(ra.get('response_text') or '')[:96]}")
            print(f"    B: {(rb.get('response_text') or '')[:96]}")
        if not checks["patch shape"]:
            print(f"    A patches: {patch_shape(ra.get('story_patches'))}")
            print(f"    B patches: {patch_shape(rb.get('story_patches'))}")


if __name__ == "__main__":
    main()
