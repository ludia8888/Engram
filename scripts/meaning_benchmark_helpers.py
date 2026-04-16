from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from engram.meaning_index import normalize_query_for_meaning_cache
from engram.search_terms import query_candidate_terms
from engram.types import MeaningAnalysis, MeaningUnit, QueryMeaningPlan


@dataclass(frozen=True, slots=True)
class MeaningSearchCase:
    name: str
    query: str
    expected_entity_id: str
    description: str


class BenchmarkMeaningAnalyzer:
    def __init__(
        self,
        *,
        version: str = "benchmark-meaning-v1",
        event_units: dict[str, list[MeaningUnit]] | None = None,
        query_plans: dict[str, QueryMeaningPlan] | None = None,
    ):
        self.version = version
        self._event_units = {
            key: [self._clone_unit(unit) for unit in units]
            for key, units in (event_units or {}).items()
        }
        self._query_plans = {
            key: self._clone_plan(plan)
            for key, plan in (query_plans or {}).items()
        }
        self.plan_calls: list[str] = []

    def analyze_event(self, event):
        entity_id = str(event.data.get("id", ""))
        return MeaningAnalysis(
            units=[self._clone_unit(unit) for unit in self._event_units.get(entity_id, [])]
        )

    def plan_query(self, query: str) -> QueryMeaningPlan:
        self.plan_calls.append(query)
        plan = self._query_plans.get(query)
        if plan is None:
            terms = query_candidate_terms(query)
            return QueryMeaningPlan(
                units=[
                    MeaningUnit(
                        kind="fallback_term",
                        value=term,
                        normalized_value=normalize_query_for_meaning_cache(term),
                        confidence=1.0,
                    )
                    for term in terms
                ],
                fallback_terms=terms,
                planner_confidence=None,
            )
        return self._clone_plan(plan)

    def _clone_plan(self, plan: QueryMeaningPlan) -> QueryMeaningPlan:
        return QueryMeaningPlan(
            units=[self._clone_unit(unit) for unit in plan.units],
            fallback_terms=list(plan.fallback_terms),
            planner_confidence=plan.planner_confidence,
        )

    def _clone_unit(self, unit: MeaningUnit) -> MeaningUnit:
        return MeaningUnit(
            kind=unit.kind,
            value=unit.value,
            normalized_value=unit.normalized_value,
            key=unit.key,
            confidence=unit.confidence,
            metadata=dict(unit.metadata),
        )


PHRASE_TARGET_ID = "user:phrase-target"
ALIAS_TARGET_ID = "user:zzz-alias-target"
KEY_TARGET_ID = "project:key-target"


def build_meaning_search_cases() -> list[MeaningSearchCase]:
    return [
        MeaningSearchCase(
            name="protected_phrase",
            query="Busan-1499 traveler",
            expected_entity_id=PHRASE_TARGET_ID,
            description="Compound label should beat broad Busan/traveler overlap.",
        ),
        MeaningSearchCase(
            name="rare_alias",
            query="special traveler",
            expected_entity_id=ALIAS_TARGET_ID,
            description="Rare alias should beat many common facet-only distractors.",
        ),
        MeaningSearchCase(
            name="canonical_key",
            query="alpha 42",
            expected_entity_id=KEY_TARGET_ID,
            description="Canonical key / alias should beat broad alpha+42 combinations.",
        ),
    ]


def build_benchmark_meaning_analyzer() -> BenchmarkMeaningAnalyzer:
    event_units: dict[str, list[MeaningUnit]] = {
        PHRASE_TARGET_ID: [
            MeaningUnit(
                kind="protected_phrase",
                value="Busan-1499",
                normalized_value="busan-1499",
                confidence=1.0,
            ),
            MeaningUnit(
                kind="alias",
                value="Busan 1499",
                normalized_value="busan 1499",
                confidence=0.9,
            ),
            MeaningUnit(
                kind="facet",
                key="role",
                value="traveler",
                normalized_value="traveler",
                confidence=0.8,
            ),
        ],
        ALIAS_TARGET_ID: [
            MeaningUnit(
                kind="alias",
                value="special traveler",
                normalized_value="special traveler",
                confidence=1.0,
            ),
            MeaningUnit(
                kind="facet",
                key="role",
                value="traveler",
                normalized_value="traveler",
                confidence=0.8,
            ),
            MeaningUnit(
                kind="facet",
                key="category",
                value="user",
                normalized_value="user",
                confidence=0.8,
            ),
        ],
        KEY_TARGET_ID: [
            MeaningUnit(
                kind="canonical_key",
                value="code:alpha-42",
                normalized_value="code:alpha-42",
                confidence=1.0,
            ),
            MeaningUnit(
                kind="alias",
                value="alpha 42",
                normalized_value="alpha 42",
                confidence=0.9,
            ),
            MeaningUnit(
                kind="facet",
                key="project",
                value="alpha",
                normalized_value="alpha",
                confidence=0.8,
            ),
        ],
    }
    for index in range(24):
        event_units[f"user:aaa-alias-distractor:{index}"] = [
            MeaningUnit(
                kind="facet",
                key="role",
                value="traveler",
                normalized_value="traveler",
                confidence=1.0,
            ),
            MeaningUnit(
                kind="facet",
                key="category",
                value="user",
                normalized_value="user",
                confidence=1.0,
            ),
        ]
        event_units[f"project:key-distractor:{index}"] = [
            MeaningUnit(
                kind="facet",
                key="project",
                value="alpha",
                normalized_value="alpha",
                confidence=1.0,
            ),
            MeaningUnit(
                kind="facet",
                key="floor",
                value="42",
                normalized_value="42",
                confidence=1.0,
            ),
        ]

    query_plans = {
        "Busan-1499 traveler": QueryMeaningPlan(
            units=[
                MeaningUnit(
                    kind="protected_phrase",
                    value="Busan-1499",
                    normalized_value="busan-1499",
                    confidence=1.0,
                ),
                MeaningUnit(
                    kind="facet",
                    key="role",
                    value="traveler",
                    normalized_value="traveler",
                    confidence=0.8,
                ),
            ],
            fallback_terms=["busan-1499", "busan", "1499", "traveler"],
            planner_confidence=0.96,
        ),
        "special traveler": QueryMeaningPlan(
            units=[
                MeaningUnit(
                    kind="alias",
                    value="special traveler",
                    normalized_value="special traveler",
                    confidence=1.0,
                ),
                MeaningUnit(
                    kind="facet",
                    key="role",
                    value="traveler",
                    normalized_value="traveler",
                    confidence=1.0,
                ),
                MeaningUnit(
                    kind="facet",
                    key="category",
                    value="user",
                    normalized_value="user",
                    confidence=1.0,
                ),
            ],
            fallback_terms=["special", "traveler"],
            planner_confidence=0.95,
        ),
        "alpha 42": QueryMeaningPlan(
            units=[
                MeaningUnit(
                    kind="canonical_key",
                    value="code:alpha-42",
                    normalized_value="code:alpha-42",
                    confidence=1.0,
                ),
                MeaningUnit(
                    kind="alias",
                    value="alpha 42",
                    normalized_value="alpha 42",
                    confidence=0.9,
                ),
            ],
            fallback_terms=["alpha", "42"],
            planner_confidence=0.94,
        ),
    }
    return BenchmarkMeaningAnalyzer(event_units=event_units, query_plans=query_plans)


def append_meaning_search_dataset(mem, *, filler_count: int) -> datetime:
    base = datetime(2026, 5, 1, tzinfo=UTC)

    for idx in range(filler_count):
        observed_at = base + timedelta(minutes=idx)
        mem.append(
            "entity.create",
            {
                "id": f"user:filler:{idx}",
                "type": "user",
                "attrs": {"label": f"Busan-{idx}", "role": "traveler", "tag": "traveler"},
            },
            observed_at=observed_at,
            effective_at_start=observed_at,
            time_confidence="exact",
            reason="generic filler entity",
        )

    mem.append(
        "entity.create",
        {
            "id": PHRASE_TARGET_ID,
            "type": "user",
            "attrs": {"label": "Busan-1499", "role": "traveler", "tag": "traveler"},
        },
        observed_at=base + timedelta(days=1),
        effective_at_start=base + timedelta(days=1),
        time_confidence="exact",
        reason="exact protected phrase entity",
    )

    for idx in range(24):
        observed_at = base + timedelta(days=2, minutes=idx)
        mem.append(
            "entity.create",
            {
                "id": f"user:aaa-alias-distractor:{idx}",
                "type": "user",
                "attrs": {"city": "special", "role": "traveler", "category": "user"},
            },
            observed_at=observed_at,
            effective_at_start=observed_at,
            time_confidence="exact",
            reason="common facet-only alias distractor",
        )
    mem.append(
        "entity.create",
        {
            "id": ALIAS_TARGET_ID,
            "type": "user",
            "attrs": {"label": "special-traveler", "role": "traveler", "category": "user"},
        },
        observed_at=base + timedelta(days=2, hours=1),
        effective_at_start=base + timedelta(days=2, hours=1),
        time_confidence="exact",
        reason="rare alias target entity",
    )

    for idx in range(24):
        observed_at = base + timedelta(days=3, minutes=idx)
        mem.append(
            "entity.create",
            {
                "id": f"project:key-distractor:{idx}",
                "type": "project",
                "attrs": {"project": "alpha", "floor": "42", "label": f"alpha-{idx}"},
            },
            observed_at=observed_at,
            effective_at_start=observed_at,
            time_confidence="exact",
            reason="broad alpha + 42 distractor",
        )
    mem.append(
        "entity.create",
        {
            "id": KEY_TARGET_ID,
            "type": "project",
            "attrs": {"label": "alpha-42", "project": "alpha"},
        },
        observed_at=base + timedelta(days=3, hours=1),
        effective_at_start=base + timedelta(days=3, hours=1),
        time_confidence="exact",
        reason="canonical key target entity",
    )

    return base
