from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from engram import Engram


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    label: str
    repeat: int
    total_seconds: float
    avg_ms: float
    p95_ms: float
    throughput_per_second: float


def dt(value: datetime) -> datetime:
    return value.astimezone(UTC)


def parse_entity_counts(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",")]
    counts = [int(value) for value in values if value]
    if not counts:
        raise ValueError("at least one entity count is required")
    return counts


def populate_memory(
    mem: Engram,
    *,
    entity_count: int,
    relation_stride: int = 5,
) -> datetime:
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for idx in range(entity_count):
        entity_id = f"user:{idx}"
        observed_at = dt(base + timedelta(minutes=idx))
        mem.append(
            "entity.create",
            {"id": entity_id, "type": "user", "attrs": {"city": f"Busan-{idx}", "tag": "traveler"}},
            observed_at=observed_at,
            effective_at_start=observed_at,
            time_confidence="exact",
        )
        if relation_stride > 0 and idx % relation_stride == 0:
            mem.append(
                "relation.create",
                {
                    "source": entity_id,
                    "target": f"user:{(idx + 1) % entity_count}",
                    "type": "teammate",
                    "attrs": {"squad": "engram"},
                },
                observed_at=dt(base + timedelta(minutes=idx, seconds=30)),
                effective_at_start=observed_at,
                time_confidence="exact",
            )
    return base


def measure_runs(fn, *, repeat: int, operations_per_run: int = 1, label: str) -> BenchmarkStats:
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    total = sum(samples)
    return BenchmarkStats(
        label=label,
        repeat=repeat,
        total_seconds=total,
        avg_ms=(total / repeat) * 1000,
        p95_ms=_percentile(samples, 0.95) * 1000,
        throughput_per_second=(operations_per_run * repeat / total) if total > 0 else 0.0,
    )


def print_stats(stats: BenchmarkStats) -> None:
    print(
        f"{stats.label}: avg {stats.avg_ms:.2f}ms, p95 {stats.p95_ms:.2f}ms, "
        f"total {stats.total_seconds:.4f}s / {stats.repeat} runs, "
        f"throughput {stats.throughput_per_second:.2f}/s"
    )


def wait_for(pred, *, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]
