from __future__ import annotations

import argparse
import tempfile
from datetime import timedelta
from pathlib import Path

from engram import Engram

from benchmark_common import dt, measure_runs, parse_entity_counts, populate_memory, print_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Engram search/context latency across dataset sizes.")
    parser.add_argument(
        "--entity-counts",
        default="1500,5000,10000",
        help="Comma-separated entity counts to benchmark (default: 1500,5000,10000).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="How many times to run each benchmark operation per dataset size.",
    )
    args = parser.parse_args()

    print("Search latency benchmark")
    for entity_count in parse_entity_counts(args.entity_counts):
        _run_for_entity_count(entity_count=entity_count, repeat=args.repeat)


def _run_for_entity_count(*, entity_count: int, repeat: int) -> None:
    with tempfile.TemporaryDirectory(prefix="engram-bench-search-") as tmpdir:
        root = Path(tmpdir)
        mem = Engram(user_id="bench", path=str(root))
        base = populate_memory(mem, entity_count=entity_count)
        mem.flush("index")

        print(f"\nDataset: {entity_count} entities")
        print_stats(
            measure_runs(
                lambda: mem.search(f"Busan-{entity_count - 1} traveler", k=5),
                repeat=repeat,
                label="known lexical search",
            )
        )
        print_stats(
            measure_runs(
                lambda: mem.search("travel partner", k=5),
                repeat=repeat,
                label="repeated semantic-capable search",
            )
        )
        print_stats(
            measure_runs(
                lambda: mem.search(
                    "teammate engram",
                    time_mode="valid",
                    time_window=(dt(base), dt(base + timedelta(days=3))),
                    k=5,
                ),
                repeat=repeat,
                label="valid relation-window search",
            )
        )
        print_stats(
            measure_runs(
                lambda: mem.context(f"Busan-{entity_count - 1} traveler", max_tokens=400),
                repeat=repeat,
                label="known context build",
            )
        )
        print_stats(
            measure_runs(
                lambda: mem.context(
                    "teammate engram",
                    time_mode="valid",
                    time_window=(dt(base), dt(base + timedelta(days=3))),
                    max_tokens=400,
                ),
                repeat=repeat,
                label="valid window context build",
            )
        )
        mem.close()


if __name__ == "__main__":
    main()
