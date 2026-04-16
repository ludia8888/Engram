from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from engram.server import create_app

from tests.conftest import dt


def _client(tmp_path, **kwargs) -> TestClient:
    app = create_app(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=False,
        **kwargs,
    )
    return TestClient(app)


def test_health(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["user_id"] == "alice"
        assert body["auto_flush"] is False
        assert body["worker_alive"] is None


def test_turn_round_trip(tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/turn", json={
            "user": "hello",
            "assistant": "hi",
            "observed_at": "2026-05-01T10:00:00Z",
        })
        assert resp.status_code == 200
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
        assert resp.status_code == 200
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


def test_raw_get_404(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/raw/nonexistent-id")
        assert resp.status_code == 404
