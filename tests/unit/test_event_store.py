from __future__ import annotations

from uuid import uuid4

from engram.storage.store import EventStore, open_connection
from engram.time_utils import utcnow
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


def test_event_store_assigns_unique_monotonic_seq_and_commits_related_rows(tmp_path):
    conn = open_connection(tmp_path / "engram.db")
    store = EventStore(conn)

    with store.transaction() as tx:
        first = _event(str(uuid4()), store.next_seq(tx), "entity.create", {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}})
        store.append_event(tx, first)
        store.append_event_entities(tx, first.id, [("user:alice", "subject")])
        store.mark_dirty(tx, [(str(uuid4()), "user:alice", first.recorded_at.isoformat().replace("+00:00", "Z"), None, "entity.create", first.recorded_at.isoformat().replace("+00:00", "Z"))])

        second = _event(str(uuid4()), store.next_seq(tx), "entity.update", {"id": "user:alice", "attrs": {"location": "Busan"}})
        store.append_event(tx, second)
        store.append_event_entities(tx, second.id, [("user:alice", "subject")])

    rows = conn.execute("SELECT seq FROM events ORDER BY seq").fetchall()
    assert [row[0] for row in rows] == [1, 2]
    assert store.count_dirty_ranges() == 1

