from __future__ import annotations

import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from engram import Engram


def dt(value: datetime) -> datetime:
    return value.astimezone(UTC)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="engram-bench-") as tmpdir:
        root = Path(tmpdir)
        mem = Engram(user_id="bench", path=str(root))
        base = datetime(2026, 5, 1, tzinfo=UTC)

        for idx in range(1500):
            entity_id = f"user:{idx}"
            mem.append(
                "entity.create",
                {"id": entity_id, "type": "user", "attrs": {"city": f"Busan-{idx}", "tag": "traveler"}},
                observed_at=dt(base + timedelta(minutes=idx)),
                effective_at_start=dt(base + timedelta(minutes=idx)),
                time_confidence="exact",
            )
            if idx % 5 == 0:
                mem.append(
                    "relation.create",
                    {
                        "source": entity_id,
                        "target": f"user:{(idx + 1) % 1500}",
                        "type": "teammate",
                        "attrs": {"squad": "engram"},
                    },
                    observed_at=dt(base + timedelta(minutes=idx, seconds=30)),
                    effective_at_start=dt(base + timedelta(minutes=idx)),
                    time_confidence="exact",
                )

        mem.flush("index")

        known_elapsed = _measure(
            lambda: mem.search("Busan-1499 traveler", k=5),
            repeat=20,
        )
        semantic_repeat_elapsed = _measure(
            lambda: mem.search("travel partner", k=5),
            repeat=20,
        )
        valid_relation_elapsed = _measure(
            lambda: mem.search(
                "teammate engram",
                time_mode="valid",
                time_window=(dt(base), dt(base + timedelta(days=3))),
                k=5,
            ),
            repeat=20,
        )
        known_context_elapsed = _measure(
            lambda: mem.context("Busan-1499 traveler", max_tokens=400),
            repeat=20,
        )
        valid_context_elapsed = _measure(
            lambda: mem.context(
                "teammate engram",
                time_mode="valid",
                time_window=(dt(base), dt(base + timedelta(days=3))),
                max_tokens=400,
            ),
            repeat=20,
        )

        print("Search latency benchmark")
        print(f"known lexical search: {known_elapsed:.4f}s total / 20 runs")
        print(f"repeated semantic-capable search: {semantic_repeat_elapsed:.4f}s total / 20 runs")
        print(f"valid relation-window search: {valid_relation_elapsed:.4f}s total / 20 runs")
        print(f"known context build: {known_context_elapsed:.4f}s total / 20 runs")
        print(f"valid window context build: {valid_context_elapsed:.4f}s total / 20 runs")
        mem.close()


def _measure(fn, *, repeat: int) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return time.perf_counter() - start


if __name__ == "__main__":
    main()
