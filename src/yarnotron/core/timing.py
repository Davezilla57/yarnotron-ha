from __future__ import annotations

from contextlib import contextmanager
import json
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
        print("timing " + json.dumps(payload, sort_keys=True), flush=True)


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)
