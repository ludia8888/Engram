from __future__ import annotations

import importlib

import pytest

from engram import Engram, ValidationError
from engram.time_utils import utcnow

from tests.conftest import dt


def test_append_create_update_and_get_known_time(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))

    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
        effective_at_start=dt("2026-04-29T00:00:00Z"),
        source_role="manual",
        reason="user stated they moved last week",
        time_confidence="inferred",
    )

    future = dt("2030-01-01T00:00:00Z")
    view = mem.get_known_at("user:alice", future)
    assert view is not None
    assert view.attrs == {"diet": "vegetarian", "location": "Busan"}

    current = mem.get("user:alice")
    assert current is not None
    assert current.attrs == view.attrs

    history = mem.known_history("user:alice")
    assert [(entry.attr, entry.old_value, entry.new_value) for entry in history] == [
        ("diet", None, "vegetarian"),
        ("location", None, "Busan"),
    ]

    mem.close()


def test_backdated_effective_time_does_not_affect_known_time_before_recorded_at(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    before_append = utcnow()

    view_before = mem.get_known_at("user:alice", before_append)
    assert view_before is None

    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-04-29T00:00:00Z"),
        source_role="manual",
        time_confidence="inferred",
    )

    still_before = mem.get_known_at("user:alice", before_append)
    assert still_before is None
    assert mem.get_known_at("user:alice", dt("2030-01-01T00:00:00Z")) is not None

    mem.close()


def test_get_uses_one_consistent_now_snapshot(tmp_path, monkeypatch):
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

    engram_module = importlib.import_module("engram.engram")
    calls: list[object] = []

    def fake_utcnow():
        calls.append(object())
        return dt("2030-01-01T00:00:00Z")

    monkeypatch.setattr(engram_module, "utcnow", fake_utcnow)

    current = mem.get("user:alice")

    assert current is not None
    assert current.attrs == {"diet": "vegetarian", "location": "Busan"}
    assert len(calls) == 1

    mem.close()


def test_relation_events_are_rejected_in_phase_1_runtime(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))

    with pytest.raises(ValidationError, match="planned but not implemented in Phase 1"):
        mem.append(
            "relation.create",
            {
                "source": "user:alice",
                "target": "project:engram",
                "type": "owns",
                "attrs": {},
            },
        )

    mem.close()
