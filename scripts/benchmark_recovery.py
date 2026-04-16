from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

from engram import Engram

from benchmark_common import BenchmarkStats, measure_runs, populate_memory, print_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Engram startup recovery paths.")
    parser.add_argument("--entity-count", type=int, default=2000, help="How many entities to preload into the benchmark dataset.")
    parser.add_argument("--repeat", type=int, default=3, help="How many times to run each recovery scenario.")
    args = parser.parse_args()

    print("Recovery benchmark")
    print_stats(_benchmark_snapshot_startup(args.entity_count, args.repeat))
    print_stats(_benchmark_rebuild_startup(args.entity_count, args.repeat))
    print_stats(_benchmark_semantic_backfill_startup(args.entity_count, args.repeat))


def _benchmark_snapshot_startup(entity_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_recovery_scenario(entity_count=entity_count, mode="snapshot"),
        repeat=repeat,
        label="startup with snapshot + fresh index",
    )


def _benchmark_rebuild_startup(entity_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_recovery_scenario(entity_count=entity_count, mode="rebuild"),
        repeat=repeat,
        label="startup rebuild after snapshot removal",
    )


def _benchmark_semantic_backfill_startup(entity_count: int, repeat: int) -> BenchmarkStats:
    return measure_runs(
        lambda: _run_recovery_scenario(entity_count=entity_count, mode="semantic_backfill"),
        repeat=repeat,
        label="startup with missing semantic index rows",
    )


def _run_recovery_scenario(*, entity_count: int, mode: str) -> None:
    with tempfile.TemporaryDirectory(prefix="engram-bench-recovery-") as tmpdir:
        root = Path(tmpdir)
        first = Engram(user_id="bench", path=str(root))
        populate_memory(first, entity_count=entity_count)
        first.flush("all")
        embedder_version = first.embedder.version
        db_path = first.db_path
        first.close()

        if mode == "rebuild":
            _mutate_db(db_path, "DELETE FROM snapshots")
        elif mode == "semantic_backfill":
            _mutate_db(db_path, "DELETE FROM vec_events WHERE embedder_version = ?", (embedder_version,))

        second = Engram(user_id="bench", path=str(root))
        if mode == "semantic_backfill":
            assert second.store.count_vec_events(embedder_version) > 0
        second.close()


def _mutate_db(db_path: Path, sql: str, params: tuple = ()) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
