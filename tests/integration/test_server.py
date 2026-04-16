from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from engram.meaning_index import normalize_query_for_meaning_cache
from engram.server import create_app
from engram.types import MeaningAnalysis, MeaningUnit, QueryMeaningPlan

from tests.conftest import dt


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
            key: list(units)
            for key, units in (event_units or {}).items()
        }
        self._query_plans = dict(query_plans or {})

    def analyze_event(self, event):
        entity_id = str(event.data.get("id", ""))
        return MeaningAnalysis(units=list(self._event_units.get(entity_id, [])))

    def plan_query(self, query: str) -> QueryMeaningPlan:
        plan = self._query_plans.get(query)
        if plan is None:
            raise AssertionError(f"missing query plan for {query!r}")
        return QueryMeaningPlan(
            units=list(plan.units),
            fallback_terms=list(plan.fallback_terms),
            planner_confidence=plan.planner_confidence,
        )


def _client(tmp_path, **kwargs) -> TestClient:
    app = create_app(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=False,
        **kwargs,
    )
    return TestClient(app)


def _wait_for(pred, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_health(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["user_id"] == "alice"
        assert body["auto_flush"] is False
        assert body["worker_alive"] is None


def test_health_reports_live_worker_when_auto_flush_enabled(tmp_path):
    app = create_app(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=True,
    )
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["auto_flush"] is True
        assert body["worker_alive"] is True


def test_turn_round_trip(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/turn", json={
            "user": "hello",
            "assistant": "hi",
            "observed_at": "2026-05-01T10:00:00Z",
        })
        assert resp.status_code == 201
        ack = resp.json()
        assert "turn_id" in ack
        assert ack["queued"] is True
        assert ack["observed_at"].endswith("Z")

        resp2 = client.get(f"/raw/{ack['turn_id']}")
        assert resp2.status_code == 200
        assert resp2.json()["user"] == "hello"


def test_append_and_get_entity(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        assert resp.status_code == 201
        assert "event_id" in resp.json()

        resp2 = client.get("/entity/user:alice")
        assert resp2.status_code == 200
        body = resp2.json()
        assert body["id"] == "user:alice"
        assert body["attrs"] == {"diet": "vegetarian"}
        assert body["created_recorded_at"].endswith("Z")


def test_get_entity_404(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/entity/user:nobody")
        assert resp.status_code == 404


def test_known_at_and_valid_at(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
            "observed_at": "2026-05-01T10:00:00Z",
        })

        resp = client.get("/entity/user:alice/known-at?at=2026-05-01T10:00:01Z")
        assert resp.status_code == 200
        assert resp.json()["attrs"] == {"location": "Seoul"}
        assert resp.json()["basis"] == "known"

        resp2 = client.get("/entity/user:alice/valid-at?at=2026-05-01T10:00:00Z")
        assert resp2.status_code == 200
        assert resp2.json()["unknown_attrs"] == ["location"]


def test_history(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        client.post("/append", json={
            "event_type": "entity.update",
            "data": {"id": "user:alice", "attrs": {"diet": "vegan"}},
            "observed_at": "2026-05-01T11:00:00Z",
        })

        resp = client.get("/entity/user:alice/history?time_mode=known")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) == 2
        assert entries[0]["old_value"] is None
        assert entries[0]["new_value"] == "vegetarian"
        assert entries[1]["old_value"] == "vegetarian"
        assert entries[1]["new_value"] == "vegan"


def test_relations(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "person:bob", "type": "person", "attrs": {}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        client.post("/append", json={
            "event_type": "relation.create",
            "data": {"source": "user:alice", "target": "person:bob", "type": "friend", "attrs": {}},
            "observed_at": "2026-05-01T10:00:00Z",
        })

        resp = client.get("/entity/user:alice/relations")
        assert resp.status_code == 200
        edges = resp.json()
        assert len(edges) == 1
        assert edges[0]["other_entity_id"] == "person:bob"


def test_search(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
            "observed_at": "2026-05-01T10:00:00Z",
            "reason": "moved to Busan",
        })
        client.post("/flush", json={"level": "index"})

        resp = client.get("/search?query=Busan")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert results[0]["entity_id"] == "user:alice"
        assert isinstance(results[0]["matched_axes"], list)


def test_context_returns_plain_text(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        client.post("/flush", json={"level": "index"})

        resp = client.get("/context?query=Busan")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"
        assert "Busan" in resp.text


def test_server_search_can_use_meaning_analyzer(tmp_path):
    analyzer = StaticMeaningAnalyzer(
        event_units={
            "user:zzz-precise": [
                MeaningUnit(
                    kind="alias",
                    value="special traveler",
                    normalized_value="special traveler",
                    confidence=1.0,
                )
            ],
            "user:aaa-broad": [
                MeaningUnit(
                    kind="facet",
                    key="city",
                    value="special",
                    normalized_value="special",
                    confidence=1.0,
                ),
                MeaningUnit(
                    kind="facet",
                    key="role",
                    value="traveler",
                    normalized_value="traveler",
                    confidence=1.0,
                )
            ],
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
                planner_confidence=0.96,
            )
        },
    )
    with _client(tmp_path, meaning_analyzer=analyzer) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:zzz-precise", "type": "user", "attrs": {"label": "special-traveler", "role": "traveler"}},
            "observed_at": "2026-05-01T10:00:00Z",
            "reason": "rare alias entity",
        })
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:aaa-broad", "type": "user", "attrs": {"city": "special", "role": "traveler"}},
            "observed_at": "2026-05-01T10:05:00Z",
            "reason": "common facet entity",
        })
        client.post("/flush", json={"level": "index"})

        first = client.get("/search?query=special%20traveler")
        assert first.status_code == 200
        cache_key = normalize_query_for_meaning_cache("special traveler")
        assert _wait_for(
            lambda: client.app.state.engram.store.load_query_meaning_cache(
                cache_key,
                analyzer.version,
            )
            is not None
        )

        second = client.get("/search?query=special%20traveler")
        assert second.status_code == 200
        assert second.json()[0]["entity_id"] == "user:zzz-precise"


def test_flush_returns_204(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/flush", json={"level": "projection"})
        assert resp.status_code == 204


def test_validation_error_returns_400(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/append", json={
            "event_type": "invalid.type",
            "data": {},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        assert resp.status_code == 400
        assert "detail" in resp.json()


def test_missing_required_field_returns_422(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/turn", json={
            "user": "hello",
            "observed_at": "2026-05-01T10:00:00Z",
        })
        assert resp.status_code == 422


def test_raw_get_404(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/raw/nonexistent-id")
        assert resp.status_code == 404


def test_search_rejects_non_positive_k(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/search?query=test&k=0")
        assert resp.status_code == 422

        resp2 = client.get("/search?query=test&k=-1")
        assert resp2.status_code == 422


def test_append_rejects_out_of_range_confidence(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
            "observed_at": "2026-05-01T10:00:00Z",
            "confidence": 999.0,
        })
        assert resp.status_code == 422


def test_relation_history_endpoint(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "person:bob", "type": "person", "attrs": {}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        client.post("/append", json={
            "event_type": "relation.create",
            "data": {"source": "user:alice", "target": "person:bob", "type": "friend", "attrs": {"since": "2026"}},
            "observed_at": "2026-05-01T10:00:00Z",
        })

        resp = client.get("/entity/user:alice/relation-history")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) == 1
        assert entries[0]["relation_type"] == "friend"
        assert entries[0]["action"] == "create"


def test_rebuild_projection_endpoint(tmp_path):
    with _client(tmp_path) as client:
        client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"v": 1}},
            "observed_at": "2026-05-01T10:00:00Z",
        })

        resp = client.post("/rebuild-projection", json={"mode": "dirty"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["rebuilt_owner_count"] >= 1
        assert body["dirty_owner_count_after"] == 0


def test_search_rejects_reversed_time_window(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/search?query=test&time_window_start=2026-05-02T00:00:00Z&time_window_end=2026-05-01T00:00:00Z")
        assert resp.status_code == 400
        assert "before" in resp.json()["detail"]


def test_search_rejects_partial_time_window(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/search?query=test&time_window_start=2026-05-01T00:00:00Z")
        assert resp.status_code == 400
        assert "both be provided" in resp.json()["detail"]


def test_context_rejects_reversed_time_window(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/context?query=test&time_window_start=2026-05-02T00:00:00Z&time_window_end=2026-05-01T00:00:00Z")
        assert resp.status_code == 400
        assert "before" in resp.json()["detail"]


def test_relations_reject_partial_time_window(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/entity/user:alice/relations?time_mode=valid&time_window_end=2026-05-01T00:00:00Z")
        assert resp.status_code == 400
        assert "both be provided" in resp.json()["detail"]


def test_reprocess_endpoint_propagates_validation_error(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/reprocess", json={"from_turn_id": "missing-turn"})
        assert resp.status_code == 400
        assert "turn_id not found" in resp.json()["detail"]


def test_append_auto_flush_eventually_updates_derived_state(tmp_path):
    app = create_app(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=True,
    )
    with TestClient(app) as client:
        resp = client.post("/append", json={
            "event_type": "entity.create",
            "data": {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
            "observed_at": "2026-05-01T10:00:00Z",
        })
        assert resp.status_code == 201

        mem = client.app.state.engram
        assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=5.0)
        assert _wait_for(lambda: mem.store.count_vec_events(mem.embedder.version) >= 1, timeout=5.0)
