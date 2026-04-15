from __future__ import annotations

from engram import Engram
from engram.time_utils import utcnow

from tests.conftest import dt


def test_known_replay_uses_recorded_time_not_effective_time(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    before = utcnow()

    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-04-29T00:00:00Z"),
        source_role="manual",
        time_confidence="inferred",
    )

    assert mem.get_known_at("user:alice", before) is None
    after = mem.get_known_at("user:alice", dt("2030-01-01T00:00:00Z"))
    assert after is not None
    assert after.attrs["location"] == "Busan"
    mem.close()


def test_known_history_recomputes_diff_without_changed_from(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-02T10:00:00Z"),
    )

    history = mem.known_history("user:alice", attr="location")
    assert [(entry.old_value, entry.new_value) for entry in history] == [
        (None, "Seoul"),
        ("Seoul", "Busan"),
    ]
    mem.close()
