from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .search_terms import event_search_terms, query_candidate_terms
from .time_utils import to_rfc3339, utcnow
from .types import Event, MeaningAnalysis, MeaningAnalysisRun, MeaningUnit, QueryMeaningPlan


class MeaningAnalyzer(Protocol):
    version: str

    def analyze_event(self, event: Event) -> MeaningAnalysis: ...

    def plan_query(self, query: str) -> QueryMeaningPlan: ...


@dataclass(slots=True)
class NullMeaningAnalyzer:
    version: str = "meaning-null-v1"

    def analyze_event(self, event: Event) -> MeaningAnalysis:
        units = [
            MeaningUnit(
                kind="fallback_term",
                value=term,
                normalized_value=term,
                confidence=1.0,
            )
            for term in event_search_terms(event)
        ]
        return MeaningAnalysis(units=units)

    def plan_query(self, query: str) -> QueryMeaningPlan:
        terms = query_candidate_terms(query)
        return QueryMeaningPlan(
            units=[
                MeaningUnit(
                    kind="fallback_term",
                    value=term,
                    normalized_value=term,
                    confidence=1.0,
                )
                for term in terms
            ],
            fallback_terms=terms,
            planner_confidence=1.0 if terms else None,
        )


class MeaningIndexer:
    def __init__(self, store, analyzer: MeaningAnalyzer):
        self.store = store
        self.analyzer = analyzer

    def index_missing(self) -> int:
        events = self.store.events_missing_search_units(self.analyzer.version)
        if not events:
            return 0

        processed_at = utcnow()
        unit_rows: list[tuple[str, str, str, str | None, str, str, float | None, str | None]] = []
        run_rows: list[MeaningAnalysisRun] = []

        for event in events:
            try:
                analysis = self.analyzer.analyze_event(event)
            except Exception as exc:
                run_rows.append(
                    MeaningAnalysisRun(
                        event_id=event.id,
                        analyzer_version=self.analyzer.version,
                        processed_at=processed_at,
                        status="FAILED",
                        error=str(exc),
                        unit_count=0,
                    )
                )
                continue

            for unit in analysis.units:
                unit_rows.append(
                    (
                        event.id,
                        self.analyzer.version,
                        unit.kind,
                        unit.key,
                        unit.value,
                        unit.normalized_value,
                        unit.confidence,
                        json.dumps(unit.metadata, ensure_ascii=False, sort_keys=True) if unit.metadata else None,
                    )
                )
            run_rows.append(
                MeaningAnalysisRun(
                    event_id=event.id,
                    analyzer_version=self.analyzer.version,
                    processed_at=processed_at,
                    status="SUCCEEDED",
                    error=None,
                    unit_count=len(analysis.units),
                )
            )

        with self.store.transaction() as tx:
            self.store.replace_event_search_units(tx, self.analyzer.version, unit_rows)
            self.store.append_meaning_analysis_runs(tx, run_rows)
        return len(run_rows)
