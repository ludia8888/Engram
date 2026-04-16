from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from engram import Engram
from engram.types import ExtractedEvent, QueueItem

from benchmark_common import BenchmarkStats, dt, measure_runs, print_stats, wait_for


class BenchmarkExtractor:
    version = "benchmark-extractor-v1"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        return [
            ExtractedEvent(
                type="entity.update",
                data={"id": "user:bench", "attrs": {f"turn_{item.turn_id[-6:]}": True}},
                source_role="user",
                confidence=0.9,
                reason="benchmark turn",
                time_confidence="exact",
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Engram write-path throughput.")
    parser.add_argument("--event-count", type=int, default=1000, help="How many append events to write per run.")
    parser.add_argument("--turn-count", type=int, default=500, help="How many turns to write per run.")
    parser.add_argument("--repeat", type=int, default=3, help="How many times to run each benchmark scenario.")
    args = parser.parse_args()

    print("Write pipeline benchmark")
    print_stats(_benchmark_append_sync(args.event_count, args.repeat))
    print_stats(_benchmark_append_with_flush(args.event_count, args.repeat))
    print_stats(_benchmark_append_auto_flush(args.event_count, args.repeat))
    print_stats(_benchmark_turn_with_manual_flush(args.turn_count, args.repeat))
    print_stats(_benchmark_turn_auto_flush(args.turn_count, args.repeat))


def _benchmark_append_sync(event_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_append_scenario(event_count=event_count, auto_flush=False, flush_all=False),
        repeat=repeat,
        operations_per_run=event_count,
        label="append only (auto_flush=False)",
    )


def _benchmark_append_with_flush(event_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_append_scenario(event_count=event_count, auto_flush=False, flush_all=True),
        repeat=repeat,
        operations_per_run=event_count,
        label="append + flush(all) (auto_flush=False)",
    )


def _benchmark_append_auto_flush(event_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_append_scenario(event_count=event_count, auto_flush=True, flush_all=False),
        repeat=repeat,
        operations_per_run=event_count,
        label="append end-to-end (auto_flush=True)",
    )


def _benchmark_turn_with_manual_flush(turn_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_turn_scenario(turn_count=turn_count, auto_flush=False),
        repeat=repeat,
        operations_per_run=turn_count,
        label="turn + flush(all) (auto_flush=False)",
    )


def _benchmark_turn_auto_flush(turn_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_turn_scenario(turn_count=turn_count, auto_flush=True),
        repeat=repeat,
        operations_per_run=turn_count,
        label="turn end-to-end (auto_flush=True)",
    )


def _run_append_scenario(*, event_count: int, auto_flush: bool, flush_all: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="engram-bench-write-") as tmpdir:
        mem = Engram(user_id="bench", path=str(Path(tmpdir)), auto_flush=auto_flush)
        base = datetime(2026, 5, 1, tzinfo=UTC)
        for idx in range(event_count):
            observed_at = dt(base + timedelta(seconds=idx))
            mem.append(
                "entity.create",
                {"id": f"item:{idx}", "type": "item", "attrs": {"n": idx, "tag": "bench"}},
                observed_at=observed_at,
                effective_at_start=observed_at,
                time_confidence="exact",
            )
        if flush_all:
            mem.flush("all")
        elif auto_flush:
            assert wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=20.0)
            assert wait_for(lambda: mem.store.load_latest_snapshot() is not None, timeout=20.0)
            assert wait_for(lambda: mem.store.count_vec_events(mem.embedder.version) >= event_count, timeout=20.0)
        mem.close()


def _run_turn_scenario(*, turn_count: int, auto_flush: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="engram-bench-turn-") as tmpdir:
        mem = Engram(
            user_id="bench",
            path=str(Path(tmpdir)),
            extractor=BenchmarkExtractor(),
            auto_flush=auto_flush,
        )
        mem.append(
            "entity.create",
            {"id": "user:bench", "type": "user", "attrs": {"role": "benchmark"}},
            observed_at=dt(datetime(2026, 5, 1, tzinfo=UTC)),
        )
        base = datetime(2026, 5, 1, 12, tzinfo=UTC)
        for idx in range(turn_count):
            observed_at = dt(base + timedelta(seconds=idx))
            mem.turn(
                user=f"turn {idx}",
                assistant="ok",
                observed_at=observed_at,
            )
        if auto_flush:
            assert wait_for(lambda: mem.store.count_extraction_runs() >= turn_count, timeout=20.0)
            assert wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=20.0)
        else:
            mem.flush("all")
        mem.close()


if __name__ == "__main__":
    main()
