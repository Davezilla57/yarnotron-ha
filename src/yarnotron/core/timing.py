from __future__ import annotations

from contextlib import contextmanager
import json
import os
import time
from typing import Iterator


class LatencyTimer:
    def __init__(self, component: str, **metadata: object) -> None:
        self.component = component
        self.metadata = {key: value for key, value in metadata.items() if value is not None}
        self.started_at = time.perf_counter()
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.timings[f"{name}_ms"] = elapsed_ms(started_at)

    def add_metadata(self, **metadata: object) -> None:
        self.metadata.update({key: value for key, value in metadata.items() if value is not None})

    def log(self) -> None:
        payload = {
            "component": self.component,
            **self.metadata,
            **self.timings,
            "total_ms": elapsed_ms(self.started_at),
        }
        line = json.dumps(payload, sort_keys=True)
        print("timing " + line, flush=True)
        _append_to_data_volume(line)


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


_MAX_TIMING_FILE = 5_000_000


def _append_to_data_volume(line: str) -> None:
    """Keep a copy of the timing record (numbers and ids only, never words) on the data volume.

    Container logs vanish when the container is rebuilt; this file survives it. One generation is kept.
    """
    folder = os.environ.get("YARNOTRON_DATA_DIR")
    if not folder:
        return
    try:
        path = os.path.join(folder, "timing.jsonl")
        if os.path.exists(path) and os.path.getsize(path) > _MAX_TIMING_FILE:
            os.replace(path, path + ".1")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **json.loads(line)}) + "\n")
    except OSError:
        pass
