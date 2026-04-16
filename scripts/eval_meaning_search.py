from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from engram import Engram

from meaning_benchmark_helpers import (
    append_meaning_search_dataset,
    build_benchmark_meaning_analyzer,
    build_meaning_search_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs meaning-aware retrieval quality on curated cases."
    )
    parser.add_argument(
        "--filler-count",
        type=int,
        default=1500,
        help="How many generic filler entities to preload before evaluation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="How many search results to inspect per query.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the evaluation report.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="engram-eval-meaning-") as tmpdir:
        root = Path(tmpdir)
        baseline = Engram(user_id="eval", path=str(root / "baseline"))
        append_meaning_search_dataset(baseline, filler_count=args.filler_count)
        baseline.flush("index")

        meaning = Engram(
            user_id="eval",
            path=str(root / "meaning"),
            meaning_analyzer=build_benchmark_meaning_analyzer(),
        )
        append_meaning_search_dataset(meaning, filler_count=args.filler_count)
        meaning.flush("index")

        report = _build_eval_report(baseline=baseline, meaning=meaning, k=args.k)
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            _print_eval_report(report)

        baseline.close()
        meaning.close()


def _build_eval_report(*, baseline: Engram, meaning: Engram, k: int) -> dict:
    cases = build_meaning_search_cases()
    baseline_top1 = 0
    meaning_top1 = 0
    baseline_rr_total = 0.0
    meaning_rr_total = 0.0
    case_reports: list[dict] = []
    for case in cases:
        baseline_results = baseline.search(case.query, k=k)
        meaning_results = meaning.search(case.query, k=k)

        baseline_rr = _reciprocal_rank(baseline_results, case.expected_entity_id)
        meaning_rr = _reciprocal_rank(meaning_results, case.expected_entity_id)
        baseline_rr_total += baseline_rr
        meaning_rr_total += meaning_rr
        if baseline_rr == 1.0:
            baseline_top1 += 1
        if meaning_rr == 1.0:
            meaning_top1 += 1
        case_reports.append(
            {
                "name": case.name,
                "description": case.description,
                "query": case.query,
                "expected_entity_id": case.expected_entity_id,
                "baseline_top3": [_format_result(item) for item in baseline_results[:3]],
                "meaning_top3": [_format_result(item) for item in meaning_results[:3]],
                "baseline_rr": round(baseline_rr, 6),
                "meaning_rr": round(meaning_rr, 6),
            }
        )

    case_count = len(cases)
    return {
        "case_count": case_count,
        "cases": case_reports,
        "summary": {
            "baseline_top1": baseline_top1,
            "meaning_top1": meaning_top1,
            "baseline_top1_rate": baseline_top1 / case_count,
            "meaning_top1_rate": meaning_top1 / case_count,
            "baseline_mrr": baseline_rr_total / case_count,
            "meaning_mrr": meaning_rr_total / case_count,
        },
    }


def _print_eval_report(report: dict) -> None:
    print("Meaning-aware retrieval eval")
    print(f"Cases: {report['case_count']}")
    for case in report["cases"]:
        print(f"\n[{case['name']}] {case['description']}")
        print(f"query: {case['query']}")
        print(f"expected: {case['expected_entity_id']}")
        print("baseline top3: " + ", ".join(case["baseline_top3"]))
        print("meaning  top3: " + ", ".join(case["meaning_top3"]))
        print(
            f"baseline RR={case['baseline_rr']:.3f} | "
            f"meaning RR={case['meaning_rr']:.3f}"
        )
    summary = report["summary"]
    print("\nSummary")
    print(
        f"baseline top1={summary['baseline_top1']}/{report['case_count']} "
        f"({summary['baseline_top1_rate']:.1%}), "
        f"MRR={summary['baseline_mrr']:.3f}"
    )
    print(
        f"meaning  top1={summary['meaning_top1']}/{report['case_count']} "
        f"({summary['meaning_top1_rate']:.1%}), "
        f"MRR={summary['meaning_mrr']:.3f}"
    )


def _format_result(result) -> str:
    return f"{result.entity_id} ({result.score:.3f})"


def _reciprocal_rank(results, expected_entity_id: str) -> float:
    for index, result in enumerate(results, start=1):
        if result.entity_id == expected_entity_id:
            return 1.0 / index
    return 0.0


if __name__ == "__main__":
    main()
