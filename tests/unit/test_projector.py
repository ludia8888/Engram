from __future__ import annotations

from uuid import uuid4

import pytest

from engram.projector import Projector
from engram.storage.store import EventStore, open_connection
from engram.time_utils import to_rfc3339, utcnow
from engram.types import Event

from tests.conftest import dt


def _event(event_id: str, seq: int, event_type: str, data: dict) -> Event:
    now = utcnow()
    return Event(
        id=event_id,
        seq=seq,
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=None,
        effective_at_end=None,
        recorded_at=now,
        type=event_type,
        data=data,
        extraction_run_id=None,
        source_turn_id=None,
        source_role="manual",
        confidence=None,
        reason=None,
        time_confidence="unknown",
        caused_by=None,
        schema_version=1,
    )


def _mark_dirty(store: EventStore, tx, event: Event) -> None:
    store.mark_dirty(
        tx,
        [
            (
                str(uuid4()),
                event.data["id"],
                to_rfc3339(event.recorded_at),
                None,
                event.type,
                to_rfc3339(event.recorded_at),
            )
        ],
    )


def test_rebuild_dirty_materializes_latest_state_once_per_owner(tmp_path):
    conn = open_connection(tmp_path / "engram.db")
    store = EventStore(conn)
    projector = Projector(store)

    with store.transaction() as tx:
        first = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        )
        store.append_event(tx, first)
        store.append_event_entities(tx, first.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, first)

        second = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.update",
            {"id": "user:alice", "attrs": {"location": "Busan"}},
        )
        store.append_event(tx, second)
        store.append_event_entities(tx, second.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, second)

    rebuilt = projector.rebuild_dirty()

    assert rebuilt == 1
    assert dict(projector.current_snapshot())["user:alice"].attrs == {
        "diet": "vegetarian",
        "location": "Busan",
    }
    assert store.count_dirty_ranges() == 0


def test_rebuild_dirty_removes_deleted_entity_from_snapshot(tmp_path):
    conn = open_connection(tmp_path / "engram.db")
    store = EventStore(conn)
    projector = Projector(store)

    with store.transaction() as tx:
        created = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        )
        store.append_event(tx, created)
        store.append_event_entities(tx, created.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, created)

    projector.rebuild_dirty()
    assert "user:alice" in projector.current_snapshot()

    with store.transaction() as tx:
        deleted = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.delete",
            {"id": "user:alice"},
        )
        store.append_event(tx, deleted)
        store.append_event_entities(tx, deleted.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, deleted)

    projector.rebuild_dirty()

    assert "user:alice" not in projector.current_snapshot()
    assert store.count_dirty_ranges() == 0


def test_rebuild_dirty_keeps_old_snapshot_and_dirty_rows_if_rebuild_fails(tmp_path, monkeypatch):
    conn = open_connection(tmp_path / "engram.db")
    store = EventStore(conn)
    projector = Projector(store)

    with store.transaction() as tx:
        created = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        )
        store.append_event(tx, created)
        store.append_event_entities(tx, created.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, created)

    projector.rebuild_dirty()
    before_snapshot = projector.current_snapshot()

    with store.transaction() as tx:
        update = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.update",
            {"id": "user:alice", "attrs": {"location": "Busan"}},
        )
        store.append_event(tx, update)
        store.append_event_entities(tx, update.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, update)

        created_bob = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:bob", "type": "user", "attrs": {"diet": "vegan"}},
        )
        store.append_event(tx, created_bob)
        store.append_event_entities(tx, created_bob.id, [("user:bob", "subject")])
        _mark_dirty(store, tx, created_bob)

    original_materialize = store.materialize_current_entity

    def fail_for_bob(entity_id: str):
        if entity_id == "user:bob":
            raise RuntimeError("simulated rebuild failure")
        return original_materialize(entity_id)

    monkeypatch.setattr(store, "materialize_current_entity", fail_for_bob)

    with pytest.raises(RuntimeError, match="simulated rebuild failure"):
        projector.rebuild_dirty()

    assert projector.current_snapshot() is before_snapshot
    assert dict(projector.current_snapshot()) == dict(before_snapshot)
    assert projector.current_snapshot()["user:alice"].attrs == {"diet": "vegetarian"}
    assert store.count_dirty_ranges() == 2


def test_rebuild_dirty_swaps_snapshot_reference_only_after_success(tmp_path):
    conn = open_connection(tmp_path / "engram.db")
    store = EventStore(conn)
    projector = Projector(store)

    with store.transaction() as tx:
        created = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        )
        store.append_event(tx, created)
        store.append_event_entities(tx, created.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, created)

    projector.rebuild_dirty()
    first_snapshot = projector.current_snapshot()

    with store.transaction() as tx:
        update = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.update",
            {"id": "user:alice", "attrs": {"location": "Busan"}},
        )
        store.append_event(tx, update)
        store.append_event_entities(tx, update.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, update)

    projector.rebuild_dirty()
    second_snapshot = projector.current_snapshot()

    assert second_snapshot is not first_snapshot
    assert second_snapshot["user:alice"].attrs == {
        "diet": "vegetarian",
        "location": "Busan",
    }


def test_rebuild_all_builds_snapshot_from_canonical_state_and_clears_dirty_ranges(tmp_path):
    conn = open_connection(tmp_path / "engram.db")
    store = EventStore(conn)
    projector = Projector(store)

    with store.transaction() as tx:
        alice = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        )
        store.append_event(tx, alice)
        store.append_event_entities(tx, alice.id, [("user:alice", "subject")])
        _mark_dirty(store, tx, alice)

        bob = _event(
            str(uuid4()),
            store.next_seq(tx),
            "entity.create",
            {"id": "user:bob", "type": "user", "attrs": {"diet": "vegan"}},
        )
        store.append_event(tx, bob)
        store.append_event_entities(tx, bob.id, [("user:bob", "subject")])
        _mark_dirty(store, tx, bob)

    rebuilt = projector.rebuild_all()

    snapshot = projector.current_snapshot()
    assert rebuilt == 2
    assert snapshot["user:alice"].attrs == {"diet": "vegetarian"}
    assert snapshot["user:bob"].attrs == {"diet": "vegan"}
    assert store.count_dirty_ranges() == 0
