from __future__ import annotations

from engram import Engram

from tests.conftest import dt


def test_flush_projection_rebuilds_snapshot_and_clears_dirty_ranges(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
    )

    future = dt("2030-01-01T00:00:00Z")
    before_get = mem.get("user:alice")
    before_known = mem.get_known_at("user:alice", future)

    assert mem.store.count_dirty_ranges() == 2
    assert dict(mem.projector.current_snapshot()) == {}

    mem.flush("projection")

    snapshot = mem.projector.current_snapshot()
    assert mem.store.count_dirty_ranges() == 0
    assert snapshot["user:alice"].attrs == {"diet": "vegetarian", "location": "Busan"}
    assert mem.get("user:alice") == before_get
    assert mem.get_known_at("user:alice", future) == before_known

    mem.close()


def test_flush_canonical_with_default_extractor_does_not_create_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.turn(
        user="지난주에 부산으로 이사했어",
        assistant="알겠어, 부산 기준으로 정리할게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    mem.flush("canonical")
    assert mem.store.count_events() == 0
    assert mem.store.count_extraction_runs() == 1
    assert mem.queue.qsize() == 0

    mem.flush("projection")
    assert mem.store.count_events() == 0
    assert dict(mem.projector.current_snapshot()) == {}

    mem.close()


def test_flush_all_runs_canonical_projection_and_index_in_sequence(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.turn(
        user="나는 부산에 살아",
        assistant="알겠어.",
        observed_at=dt("2026-05-01T11:00:00Z"),
    )

    assert mem.store.count_dirty_ranges() == 1
    assert mem.queue.qsize() == 1

    mem.flush("all")

    assert mem.queue.qsize() == 0
    assert mem.store.count_dirty_ranges() == 0
    assert mem.store.count_extraction_runs() == 1
    snapshot = mem.projector.current_snapshot()
    assert "user:alice" in snapshot
    assert mem.store.count_vec_events() >= 1

    mem.close()
