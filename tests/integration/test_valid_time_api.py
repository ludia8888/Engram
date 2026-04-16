from __future__ import annotations

from engram import Engram

from tests.conftest import dt


def test_get_valid_at_replays_effective_time_locally(tmp_path):
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
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
    )

    before_move = mem.get_valid_at("user:alice", dt("2026-05-05T12:00:00Z"))
    after_move = mem.get_valid_at("user:alice", dt("2026-05-09T12:00:00Z"))

    assert before_move is not None
    assert before_move.attrs == {"location": "Seoul"}
    assert before_move.unknown_attrs == []

    assert after_move is not None
    assert after_move.attrs == {"location": "Busan"}
    assert after_move.unknown_attrs == []

    mem.close()


def test_get_valid_at_pushes_unknown_effective_time_into_unknown_attrs(tmp_path):
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
    )

    view = mem.get_valid_at("user:alice", dt("2026-05-15T12:00:00Z"))

    assert view is not None
    assert view.attrs == {"diet": "vegetarian"}
    assert view.unknown_attrs == ["location"]

    mem.close()


def test_valid_history_orders_by_effective_time_and_skips_unknown_time_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        reason="initial city",
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        reason="moved to Busan",
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-11T10:00:00Z"),
        reason="diet changed but time unknown",
        time_confidence="unknown",
    )

    history = mem.valid_history("user:alice")

    assert [entry.attr for entry in history] == ["location", "location"]
    assert history[0].new_value == "Seoul"
    assert history[1].new_value == "Busan"
    assert history[0].basis == "valid"
    assert history[1].basis == "valid"

    mem.close()


def test_get_valid_at_uses_half_open_effective_interval(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        effective_at_end=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-08T09:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
    )

    just_before = mem.get_valid_at("user:alice", dt("2026-05-07T23:59:59Z"))
    at_boundary = mem.get_valid_at("user:alice", dt("2026-05-08T00:00:00Z"))

    assert just_before is not None
    assert just_before.attrs == {"location": "Seoul"}
    assert at_boundary is not None
    assert at_boundary.attrs == {"location": "Busan"}

    mem.close()


def test_get_valid_at_returns_unknown_attrs_even_without_active_state(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        time_confidence="unknown",
    )

    view = mem.get_valid_at("user:alice", dt("2026-05-15T12:00:00Z"))

    assert view is not None
    assert view.attrs == {}
    assert view.unknown_attrs == ["location"]
    assert view.basis == "valid"

    mem.close()


def test_get_valid_at_does_not_leak_future_entity_type(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "robot", "attrs": {"model": "R1"}},
        observed_at=dt("2026-06-01T10:00:00Z"),
        effective_at_start=dt("2026-06-01T00:00:00Z"),
        time_confidence="exact",
    )

    view = mem.get_valid_at("user:alice", dt("2026-05-15T12:00:00Z"))

    assert view is not None
    assert view.entity_type == "user"
    assert view.attrs == {"location": "Seoul"}

    mem.close()
