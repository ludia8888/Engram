from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from engram import Engram
from engram.meaning_index import normalize_query_for_meaning_cache

from benchmark_common import measure_runs, parse_entity_counts, print_stats
from meaning_benchmark_helpers import (
    append_meaning_search_dataset,
    build_benchmark_meaning_analyzer,
    build_meaning_search_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure meaning-aware search latency and cache behavior."
    )
    parser.add_argument(
        "--entity-counts",
        default="1500,5000,10000",
        help="Comma-separated filler entity counts to benchmark (default: 1500,5000,10000).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="How many times to run each benchmark operation per dataset size.",
    )
    args = parser.parse_args()

    print("Meaning-aware search benchmark")
    for entity_count in parse_entity_counts(args.entity_counts):
        _run_for_entity_count(entity_count=entity_count, repeat=args.repeat)


def _run_for_entity_count(*, entity_count: int, repeat: int) -> None:
    cases = build_meaning_search_cases()
    phrase_case = cases[0]
    alias_case = cases[1]

    with tempfile.TemporaryDirectory(prefix="engram-bench-meaning-") as tmpdir:
        root = Path(tmpdir)
        baseline = Engram(user_id="bench", path=str(root / "baseline"))
        append_meaning_search_dataset(baseline, filler_count=entity_count)
        baseline.flush("index")

        analyzer = build_benchmark_meaning_analyzer()
        meaning = Engram(
            user_id="bench",
            path=str(root / "meaning"),
            meaning_analyzer=analyzer,
        )
        append_meaning_search_dataset(meaning, filler_count=entity_count)
        meaning.flush("index")

        print(f"\nDataset: filler_count={entity_count}")
        _print_quality_preview(baseline, meaning)

        print_stats(
            measure_runs(
                lambda: baseline.search(phrase_case.query, k=5),
                repeat=repeat,
                label="baseline lexical search",
            )
        )
        print_stats(
            measure_runs(
                lambda: meaning.search(phrase_case.query, k=5),
                repeat=repeat,
                label="meaning-aware warm search",
            )
        )
        print_stats(
            measure_runs(
                lambda: _run_cold_search(meaning, phrase_case.query),
                repeat=repeat,
                label="meaning-aware cold planner search",
            )
        )
        print_stats(
            measure_runs(
                lambda: baseline.context(phrase_case.query, max_tokens=300),
                repeat=repeat,
                label="baseline context build",
            )
        )
        print_stats(
            measure_runs(
                lambda: meaning.context(phrase_case.query, max_tokens=300),
                repeat=repeat,
                label="meaning-aware context build",
            )
        )
        print_stats(
            measure_runs(
                lambda: meaning.search(alias_case.query, k=5),
                repeat=repeat,
                label="meaning-aware rare-alias search",
            )
        )

        baseline.close()
        meaning.close()


def _run_cold_search(mem: Engram, query: str) -> None:
    with mem.store.transaction() as tx:
        mem.store.clear_query_meaning_cache(
            tx,
            analyzer_version=mem.meaning_analyzer.version,
            normalized_query=normalize_query_for_meaning_cache(query),
        )
    mem.search(query, k=5)


def _print_quality_preview(baseline: Engram, meaning: Engram) -> None:
    for case in build_meaning_search_cases():
        baseline_top = _top_entity_id(baseline.search(case.query, k=3))
        meaning_top = _top_entity_id(meaning.search(case.query, k=3))
        print(
            f"- {case.name}: expected={case.expected_entity_id} | "
            f"baseline_top1={baseline_top} | meaning_top1={meaning_top}"
        )


def _top_entity_id(results) -> str:
    if not results:
        return "<none>"
    return results[0].entity_id


if __name__ == "__main__":
    main()
