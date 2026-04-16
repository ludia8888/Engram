from __future__ import annotations

import threading
import time
import types

from engram import Engram, OpenAIMeaningAnalyzer
import engram.engram as engram_module
from engram.meaning_index import normalize_query_for_meaning_cache
import engram.openai_meaning_analyzer as openai_meaning_module
import engram.retrieval as retrieval_module
from engram.types import MeaningAnalysis, MeaningUnit, QueryMeaningPlan

from tests.conftest import dt


def _wait_for(pred, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


class StaticMeaningAnalyzer:
    def __init__(
        self,
        *,
        version: str = "meaning-static-v1",
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
            units=[
                self._clone_unit(unit)
                for unit in self._event_units.get(entity_id, [])
            ]
        )

    def plan_query(self, query: str) -> QueryMeaningPlan:
        self.plan_calls.append(query)
        plan = self._query_plans.get(query)
        if plan is None:
            raise AssertionError(f"missing query plan for {query!r}")
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


class BlockingMeaningAnalyzer(StaticMeaningAnalyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = threading.Event()
        self.release = threading.Event()

    def plan_query(self, query: str) -> QueryMeaningPlan:
        self.started.set()
        self.release.wait(timeout=5.0)
        return super().plan_query(query)


def test_search_returns_entity_seeded_by_matching_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="user said they are vegetarian",
        confidence=0.91,
        source_role="user",
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
        reason="user said they moved to Busan",
        confidence=0.95,
        source_role="user",
        time_confidence="exact",
    )
    mem.append(
        "entity.create",
        {"id": "project:trip", "type": "project", "attrs": {"destination": "Tokyo"}},
        observed_at=dt("2026-05-01T12:00:00Z"),
        reason="trip planning started",
        confidence=0.7,
        source_role="manual",
        time_confidence="exact",
    )

    results = mem.search("Busan vegetarian", k=5)

    assert results
    assert results[0].entity_id == "user:alice"
    assert results[0].time_basis == "known"
    assert "entity" in results[0].matched_axes
    assert len(results[0].supporting_event_ids) >= 2

    mem.close()


def test_search_respects_known_time_window(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-01T10:00:00Z"))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="initial city",
        source_role="manual",
    )
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-10T10:00:00Z"))
    moved_event_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        reason="moved to Busan",
        source_role="manual",
    )

    early = mem.search(
        "Busan",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-05T00:00:00Z")),
    )
    late = mem.search(
        "Busan",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
    )

    assert early == []
    assert late[0].entity_id == "user:alice"
    assert "temporal" in late[0].matched_axes
    assert moved_event_id in late[0].supporting_event_ids

    mem.close()


def test_search_normalizes_korean_particles_for_lexical_match(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-10T10:00:00Z"))
    moved_event_id = mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "부산"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        reason="사용자가 부산으로 이사했다고 말했다",
        source_role="user",
    )

    results = mem.search(
        "부산에서 뭐 먹지",
        k=5,
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
    )

    assert results
    assert results[0].entity_id == "user:alice"
    assert moved_event_id in results[0].supporting_event_ids

    mem.close()


def test_context_builds_known_time_memory_summary_and_optional_raw(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    ack = mem.turn(
        user="지난주에 부산으로 이사했고 나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "채식주의"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="사용자가 본인이 채식주의자라고 말했다",
        confidence=0.91,
        source_role="user",
        source_turn_id=ack.turn_id,
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "부산"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="사용자가 지난주에 부산으로 이사했다고 말했다",
        confidence=0.95,
        source_role="user",
        source_turn_id=ack.turn_id,
        time_confidence="inferred",
    )

    text = mem.context(
        "부산에서 채식 식당 추천해줘",
        max_tokens=400,
        include_raw=True,
    )

    assert "## Memory Basis" in text
    assert "## Current State" in text
    assert "## Relevant Changes" in text
    assert "## Raw Evidence" in text
    assert "user:alice" in text
    assert "부산" in text
    assert "채식주의" in text
    assert "지난주에 부산으로 이사했고 나는 채식주의자야" in text

    mem.close()


def test_search_respects_valid_time_window(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    moved_event_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
        reason="moved to Busan",
    )

    early = mem.search(
        "Busan",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-05T00:00:00Z")),
    )
    late = mem.search(
        "Busan",
        time_mode="valid",
        time_window=(dt("2026-05-08T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
    )

    assert early == []
    assert late[0].entity_id == "user:alice"
    assert late[0].time_basis == "valid"
    assert "temporal" in late[0].matched_axes
    assert moved_event_id in late[0].supporting_event_ids

    mem.close()


def test_search_prefers_protected_phrase_meaning_hits_over_fallback_token_matches(tmp_path):
    analyzer = StaticMeaningAnalyzer(
        event_units={
            "user:precise": [
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
            "user:broad": [
                MeaningUnit(
                    kind="alias",
                    value="Busan traveler",
                    normalized_value="busan traveler",
                    confidence=0.7,
                ),
                MeaningUnit(
                    kind="facet",
                    key="role",
                    value="traveler",
                    normalized_value="traveler",
                    confidence=0.8,
                ),
            ],
        },
        query_plans={
            "Busan-1499 traveler": QueryMeaningPlan(
                units=[
                    MeaningUnit(
                        kind="protected_phrase",
                        value="Busan-1499",
                        normalized_value="busan-1499",
                        confidence=1.0,
                    ),
                ],
                fallback_terms=["busan-1499", "busan", "1499", "traveler"],
                planner_confidence=0.96,
            )
        },
    )
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:precise", "type": "user", "attrs": {"label": "Busan-1499", "role": "traveler"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="exact protected phrase entity",
    )
    mem.append(
        "entity.create",
        {"id": "user:broad", "type": "user", "attrs": {"label": "Busan", "role": "traveler"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
        reason="broad lexical overlap entity",
    )
    mem.flush("index")

    mem.search("Busan-1499 traveler", k=5)
    assert _wait_for(
        lambda: mem.store.load_query_meaning_cache(
            normalize_query_for_meaning_cache("Busan-1499 traveler"),
            analyzer.version,
        )
        is not None
    )
    results = mem.search("Busan-1499 traveler", k=5)

    assert results
    assert results[0].entity_id == "user:precise"
    assert all(result.entity_id != "user:broad" for result in results[1:])

    mem.close()


def test_search_caches_query_meaning_plan_between_calls(tmp_path):
    analyzer = StaticMeaningAnalyzer(
        event_units={
            "user:precise": [
                MeaningUnit(
                    kind="protected_phrase",
                    value="Busan-1499",
                    normalized_value="busan-1499",
                    confidence=1.0,
                )
            ]
        },
        query_plans={
            "Busan-1499": QueryMeaningPlan(
                units=[
                    MeaningUnit(
                        kind="protected_phrase",
                        value="Busan-1499",
                        normalized_value="busan-1499",
                        confidence=1.0,
                    )
                ],
                fallback_terms=["busan-1499", "busan", "1499"],
                planner_confidence=0.99,
            )
        },
    )
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:precise", "type": "user", "attrs": {"label": "Busan-1499"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="exact protected phrase entity",
    )
    mem.flush("index")

    first = mem.search("Busan-1499", k=5)
    assert _wait_for(
        lambda: mem.store.load_query_meaning_cache(
            normalize_query_for_meaning_cache("Busan-1499"),
            analyzer.version,
        )
        is not None
    )
    second = mem.search("Busan-1499", k=5)

    assert first
    assert second
    assert first[0].entity_id == "user:precise"
    assert second[0].entity_id == "user:precise"
    assert analyzer.plan_calls == ["Busan-1499"]

    mem.close()


def test_search_cache_miss_does_not_block_on_meaning_planner(tmp_path):
    analyzer = BlockingMeaningAnalyzer(
        event_units={
            "user:precise": [
                MeaningUnit(
                    kind="alias",
                    value="special traveler",
                    normalized_value="special traveler",
                    confidence=1.0,
                )
            ]
        },
        query_plans={
            "special traveler": QueryMeaningPlan(
                units=[
                    MeaningUnit(
                        kind="alias",
                        value="special traveler",
                        normalized_value="special traveler",
                        confidence=1.0,
                    )
                ],
                fallback_terms=["special", "traveler"],
                planner_confidence=0.95,
            )
        },
    )
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:precise", "type": "user", "attrs": {"label": "special-traveler"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="rare alias entity",
    )
    mem.flush("index")

    started = time.monotonic()
    first = mem.search("special traveler", k=5)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert first
    assert analyzer.started.wait(timeout=1.0)

    analyzer.release.set()
    assert _wait_for(
        lambda: mem.store.load_query_meaning_cache(
            normalize_query_for_meaning_cache("special traveler"),
            analyzer.version,
        )
        is not None
    )

    second = mem.search("special traveler", k=5)
    assert second[0].entity_id == "user:precise"

    mem.close()


def test_query_plan_cache_write_is_best_effort_when_writer_lock_is_busy(tmp_path):
    analyzer = BlockingMeaningAnalyzer(
        event_units={
            "user:precise": [
                MeaningUnit(
                    kind="alias",
                    value="special traveler",
                    normalized_value="special traveler",
                    confidence=1.0,
                )
            ]
        },
        query_plans={
            "special traveler": QueryMeaningPlan(
                units=[
                    MeaningUnit(
                        kind="alias",
                        value="special traveler",
                        normalized_value="special traveler",
                        confidence=1.0,
                    )
                ],
                fallback_terms=["special", "traveler"],
                planner_confidence=0.95,
            )
        },
    )
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:precise", "type": "user", "attrs": {"label": "special-traveler"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="rare alias entity",
    )
    mem.flush("index")

    first = mem.search("special traveler", k=5)
    assert first
    assert analyzer.started.wait(timeout=1.0)

    cache_key = normalize_query_for_meaning_cache("special traveler")
    mem.store._tx_lock.acquire()
    try:
        analyzer.release.set()
        time.sleep(0.2)
        assert (
            mem.store.load_query_meaning_cache(cache_key, analyzer.version)
            is None
        )
    finally:
        mem.store._tx_lock.release()

    assert _wait_for(lambda: len(analyzer.plan_calls) >= 1)
    second = mem.search("special traveler", k=5)
    assert second[0].entity_id == "user:precise"
    assert _wait_for(
        lambda: mem.store.load_query_meaning_cache(cache_key, analyzer.version) is not None
    )
    assert len(analyzer.plan_calls) >= 2

    mem.close()


def test_search_meaning_idf_prefers_rare_alias_over_multiple_common_facets(tmp_path):
    event_units: dict[str, list[MeaningUnit]] = {
        "user:precise": [
            MeaningUnit(
                kind="alias",
                value="special-traveler",
                normalized_value="special-traveler",
                confidence=1.0,
            )
        ]
    }
    for index in range(20):
        event_units[f"user:broad:{index}"] = [
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

    analyzer = StaticMeaningAnalyzer(
        event_units=event_units,
        query_plans={
            "special traveler": QueryMeaningPlan(
                units=[
                    MeaningUnit(
                        kind="alias",
                        value="special-traveler",
                        normalized_value="special-traveler",
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
            )
        },
    )
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:precise", "type": "user", "attrs": {"label": "special-traveler"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="rare alias entity",
    )
    for index in range(20):
        mem.append(
            "entity.create",
            {"id": f"user:broad:{index}", "type": "user", "attrs": {"role": "traveler", "category": "user"}},
            observed_at=dt(f"2026-05-01T10:{index + 1:02d}:00Z"),
            reason="common facet entity",
    )
    mem.flush("index")

    mem.search("special traveler", k=5)
    assert _wait_for(
        lambda: mem.store.load_query_meaning_cache(
            normalize_query_for_meaning_cache("special traveler"),
            analyzer.version,
        )
        is not None
    )
    results = mem.search("special traveler", k=5)

    assert results
    assert results[0].entity_id == "user:precise"

    mem.close()


def test_search_uses_openai_meaning_analyzer_end_to_end(tmp_path, monkeypatch):
    responses = [
        {
            "units": [
                {"kind": "protected_phrase", "value": "Busan-1499", "confidence": 0.98},
                {"kind": "facet", "key": "role", "value": "traveler", "confidence": 0.8},
            ]
        },
        {
            "units": [
                {"kind": "alias", "value": "Busan traveler", "confidence": 0.72},
            ]
        },
        {
            "units": [
                {"kind": "protected_phrase", "value": "Busan-1499", "confidence": 0.99},
            ],
            "planner_confidence": 0.92,
        },
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            if not responses:
                raise AssertionError("no fake OpenAI responses left")
            payload = responses.pop(0)
            import json

            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(payload, ensure_ascii=False), refusal=None),
                        finish_reason="stop",
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai_meaning_module, "_load_openai_client_class", lambda: FakeOpenAI)

    analyzer = OpenAIMeaningAnalyzer()
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:precise", "type": "user", "attrs": {"label": "Busan-1499", "role": "traveler"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="exact protected phrase entity",
    )
    mem.append(
        "entity.create",
        {"id": "user:broad", "type": "user", "attrs": {"label": "Busan", "role": "traveler"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
        reason="broad lexical overlap entity",
    )
    mem.flush("index")

    mem.search("Busan-1499 traveler", k=5)
    assert _wait_for(
        lambda: mem.store.load_query_meaning_cache(
            normalize_query_for_meaning_cache("Busan-1499 traveler"),
            analyzer.version,
        )
        is not None
    )
    results = mem.search("Busan-1499 traveler", k=5)

    assert results
    assert results[0].entity_id == "user:precise"

    mem.close()


def test_failed_meaning_analysis_is_not_retried_forever(tmp_path):
    class FailingMeaningAnalyzer:
        version = "meaning-fail-once-v1"

        def __init__(self):
            self.analyze_calls = 0

        def analyze_event(self, event):
            self.analyze_calls += 1
            raise RuntimeError("boom")

        def plan_query(self, query: str) -> QueryMeaningPlan:
            return QueryMeaningPlan(units=[], fallback_terms=["boom"], planner_confidence=None)

    analyzer = FailingMeaningAnalyzer()
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"label": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    mem.flush("index")
    mem.flush("index")

    assert analyzer.analyze_calls == 1
    status_row = mem.store._reader_conn.execute(
        "SELECT status FROM meaning_analysis_runs WHERE analyzer_version = ?",
        (analyzer.version,),
    ).fetchone()
    assert status_row is not None
    assert status_row["status"] == "FAILED"

    mem.close()


def test_query_meaning_cache_is_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval_module, "_QUERY_MEANING_CACHE_MAX_ROWS", 2)
    analyzer = StaticMeaningAnalyzer(
        query_plans={
            "query-one": QueryMeaningPlan(units=[], fallback_terms=["query-one"], planner_confidence=0.9),
            "query-two": QueryMeaningPlan(units=[], fallback_terms=["query-two"], planner_confidence=0.9),
            "query-three": QueryMeaningPlan(units=[], fallback_terms=["query-three"], planner_confidence=0.9),
        }
    )
    mem = Engram(user_id="alice", path=str(tmp_path), meaning_analyzer=analyzer)

    for query in ("query-one", "query-two", "query-three"):
        mem.search(query, k=5)
        assert _wait_for(
            lambda query=query: mem.store.load_query_meaning_cache(
                normalize_query_for_meaning_cache(query),
                analyzer.version,
            )
            is not None
        )

    assert _wait_for(
        lambda: mem.store.count_query_meaning_cache(analyzer.version) == 2
    ), f"expected 2, got {mem.store.count_query_meaning_cache(analyzer.version)}"
    assert mem.store.load_query_meaning_cache(
        normalize_query_for_meaning_cache("query-one"),
        analyzer.version,
    ) is None
    assert mem.store.load_query_meaning_cache(
        normalize_query_for_meaning_cache("query-two"),
        analyzer.version,
    ) is not None
    assert mem.store.load_query_meaning_cache(
        normalize_query_for_meaning_cache("query-three"),
        analyzer.version,
    ) is not None

    mem.close()


def test_search_valid_skips_unknown_effective_time_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        time_confidence="unknown",
        reason="location mentioned without exact effective time",
    )

    results = mem.search("Busan", time_mode="valid")

    assert results == []

    mem.close()


def test_search_valid_without_time_window_uses_current_as_of(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    moved_event_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
        reason="moved to Busan",
    )

    monkeypatch.setattr(retrieval_module, "utcnow", lambda: dt("2026-05-09T12:00:00Z"))

    results = mem.search("Busan", time_mode="valid")

    assert results
    assert results[0].entity_id == "user:alice"
    assert results[0].time_basis == "valid"
    assert moved_event_id in results[0].supporting_event_ids

    mem.close()


def test_context_builds_valid_time_memory_summary(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    ack = mem.turn(
        user="나는 채식주의자고, 부산에 있는 것 같긴 한데 정확한 날짜는 잘 모르겠어",
        assistant="좋아, 시간 확실한 정보와 불확실한 정보를 나눠서 기억해둘게.",
        observed_at=dt("2026-05-12T09:00:00Z"),
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_turn_id=ack.turn_id,
        source_role="user",
        time_confidence="exact",
        reason="diet is known to be active from May 1",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-12T09:00:00Z"),
        source_turn_id=ack.turn_id,
        source_role="user",
        time_confidence="unknown",
        reason="location was mentioned but effective time is unknown",
    )

    text = mem.context(
        "vegetarian meal ideas",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
        include_raw=True,
        max_tokens=400,
    )

    assert "## Memory Basis" in text
    assert "- mode: valid" in text
    assert "## Current State" in text
    assert "user:alice" in text
    assert "vegetarian" in text
    assert "unknown_attrs_as_of_window_end=['location']" in text
    assert "## Relevant Changes" in text
    assert "effective_at=2026-05-01T00:00:00Z" in text
    assert "## Raw Evidence" in text
    assert "나는 채식주의자고" in text

    mem.close()


def test_batch_known_helpers_match_single_entity_reads(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="moved to Busan",
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
    )
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:06:00Z"),
    )

    at = dt("2026-05-02T00:00:00Z")
    views = mem._get_known_views_at_many(["user:alice", "user:bob"], at)
    relations = mem._get_known_relations_at_many(["user:alice", "user:bob"], at)

    assert views["user:alice"] == mem.get_known_at("user:alice", at)
    assert views["user:bob"] == mem.get_known_at("user:bob", at)
    assert relations["user:alice"] == mem.get_relations("user:alice", time_mode="known", at=at)
    assert relations["user:bob"] == mem.get_relations("user:bob", time_mode="known", at=at)

    mem.close()


def test_batch_valid_helpers_match_single_entity_reads(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-05T10:00:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"team": "engram"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-05T10:06:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        time_confidence="exact",
    )

    at = dt("2026-05-06T00:00:00Z")
    views = mem._get_valid_views_at_many(["user:alice", "user:bob"], at)
    relations = mem._get_valid_relations_at_many(["user:alice", "user:bob"], at)

    assert views["user:alice"] == mem.get_valid_at("user:alice", at)
    assert views["user:bob"] == mem.get_valid_at("user:bob", at)
    assert relations["user:alice"] == mem.get_relations("user:alice", time_mode="valid", at=at)
    assert relations["user:bob"] == mem.get_relations("user:bob", time_mode="valid", at=at)

    window_relations = mem._get_valid_relations_in_window_many(
        ["user:alice", "user:bob"],
        dt("2026-05-04T00:00:00Z"),
        dt("2026-05-07T00:00:00Z"),
    )

    assert window_relations["user:alice"] == mem.get_relations(
        "user:alice",
        time_mode="valid",
        time_window=(dt("2026-05-04T00:00:00Z"), dt("2026-05-07T00:00:00Z")),
    )
    assert window_relations["user:bob"] == mem.get_relations(
        "user:bob",
        time_mode="valid",
        time_window=(dt("2026-05-04T00:00:00Z"), dt("2026-05-07T00:00:00Z")),
    )

    mem.close()
