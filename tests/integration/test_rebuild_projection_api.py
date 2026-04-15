from __future__ import annotations

import pytest

from engram import Engram, ValidationError

from tests.conftest import dt


def _append_people(mem: Engram) -> None:
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"name": "Alice"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
        source_role="manual",
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"name": "Bob"}},
        observed_at=dt("2026-05-01T09:05:00Z"),
        source_role="manual",
    )


def _append_relation(mem: Engram) -> None:
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )


def test_rebuild_projection_dirty_rebuilds_and_clears_dirty_ranges(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        source_role="manual",
    )

    result = mem.rebuild_projection()

    assert result.scope == "dirty"
    assert result.target_owner_id is None
    assert result.rebuilt_owner_count == 1
    assert result.dirty_owner_count_before == 1
    assert result.dirty_owner_count_after == 0
    assert mem.projector.current_snapshot()["user:alice"].attrs == {"diet": "vegetarian"}

    mem.close()


def test_rebuild_projection_dirty_noop_returns_zero_counts(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))

    result = mem.rebuild_projection()

    assert result.scope == "dirty"
    assert result.rebuilt_owner_count == 0
    assert result.dirty_owner_count_before == 0
    assert result.dirty_owner_count_after == 0

    mem.close()


def test_rebuild_projection_owner_refreshes_stale_snapshot_without_dirty_rows(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    _append_relation(mem)
    mem.rebuild_projection()

    mem.append(
        "entity.delete",
        {"id": "user:alice"},
        observed_at=dt("2026-05-02T10:00:00Z"),
        effective_at_start=dt("2026-05-02T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    with mem.store.transaction() as tx:
        mem.store.clear_all_dirty_ranges(tx)

    result = mem.rebuild_projection(owner_id="user:alice")

    assert result.scope == "owner"
    assert result.target_owner_id == "user:alice"
    assert result.rebuilt_owner_count == 1
    assert result.dirty_owner_count_before == 0
    assert result.dirty_owner_count_after == 0
    assert "user:alice" not in mem.projector.current_snapshot()
    assert "user:alice" not in mem.projector.current_relation_snapshot()

    mem.close()


def test_rebuild_projection_owner_only_clears_target_owner_dirty_rows(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    _append_relation(mem)
    mem.rebuild_projection()

    mem.append(
        "entity.delete",
        {"id": "user:alice"},
        observed_at=dt("2026-05-02T10:00:00Z"),
        effective_at_start=dt("2026-05-02T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    before_dirty = set(mem.store.dirty_owner_ids())
    result = mem.rebuild_projection(owner_id="user:alice")
    after_dirty = set(mem.store.dirty_owner_ids())

    assert before_dirty == {"user:alice", "user:bob"}
    assert result.scope == "owner"
    assert result.dirty_owner_count_before == 2
    assert result.dirty_owner_count_after == 1
    assert after_dirty == {"user:bob"}
    assert "user:alice" not in mem.projector.current_snapshot()
    assert "user:alice" not in mem.projector.current_relation_snapshot()

    mem.close()


def test_rebuild_projection_full_rebuilds_entire_snapshot_and_relations(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    _append_relation(mem)
    mem.rebuild_projection()

    mem.append(
        "relation.delete",
        {"source": "user:alice", "target": "user:bob", "type": "manager"},
        observed_at=dt("2026-05-03T10:00:00Z"),
        effective_at_start=dt("2026-05-03T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:bob", "attrs": {"role": "lead"}},
        observed_at=dt("2026-05-03T11:00:00Z"),
        source_role="manual",
    )
    with mem.store.transaction() as tx:
        mem.store.clear_all_dirty_ranges(tx)

    result = mem.rebuild_projection(mode="full")

    assert result.scope == "full"
    assert result.target_owner_id is None
    assert result.rebuilt_owner_count == 2
    assert result.dirty_owner_count_before == 0
    assert result.dirty_owner_count_after == 0
    assert mem.projector.current_snapshot()["user:bob"].attrs == {"name": "Bob", "role": "lead"}
    assert "user:alice" not in mem.projector.current_relation_snapshot()
    assert "user:bob" not in mem.projector.current_relation_snapshot()

    mem.close()


def test_rebuild_projection_full_rejects_owner_id(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))

    with pytest.raises(ValidationError, match="owner_id is not supported"):
        mem.rebuild_projection(mode="full", owner_id="user:alice")

    mem.close()
