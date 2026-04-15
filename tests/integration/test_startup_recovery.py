from __future__ import annotations

import pytest

from engram import Engram, QueueFullError

from tests.conftest import dt


def test_startup_catch_up_requeues_raw_turns_without_canonical_events(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    ack = first.turn(
        user="지난주에 부산으로 이사했어",
        assistant="알겠어, 부산 기준으로 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    assert second.queue.qsize() == 1
    queued = second.queue.get_nowait()
    assert queued.turn_id == ack.turn_id
    assert queued.user == "지난주에 부산으로 이사했어"

    second.close()


def test_startup_catch_up_skips_turns_already_linked_to_canonical_events(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    ack = first.turn(
        user="난 채식주의자야",
        assistant="알겠어, 식단 선호로 기억할게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.append(
        "entity.create",
        {
            "id": "user:alice",
            "type": "user",
            "attrs": {"diet": "vegetarian"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        source_turn_id=ack.turn_id,
        source_role="user",
        time_confidence="exact",
    )
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    assert second.queue.qsize() == 0

    second.close()


def test_startup_recovery_rebuilds_pending_projection_snapshot(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.append(
        "entity.create",
        {
            "id": "user:alice",
            "type": "user",
            "attrs": {"diet": "vegetarian"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    assert first.store.count_dirty_ranges() == 1
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    snapshot = second.projector.current_snapshot()
    assert second.store.count_dirty_ranges() == 0
    assert snapshot["user:alice"].attrs == {"diet": "vegetarian"}

    second.close()


def test_startup_catch_up_raises_if_queue_cannot_hold_gap(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.turn(
        user="첫 번째",
        assistant="응답 1",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.turn(
        user="두 번째",
        assistant="응답 2",
        observed_at=dt("2026-05-01T10:01:00Z"),
    )
    first.close()

    with pytest.raises(QueueFullError, match="startup catch-up could not enqueue raw turn"):
        Engram(user_id="alice", path=str(tmp_path), queue_max_size=1, queue_put_timeout=0.001)


def test_startup_failure_releases_writer_lock_and_allows_retry(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.turn(
        user="첫 번째",
        assistant="응답 1",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.turn(
        user="두 번째",
        assistant="응답 2",
        observed_at=dt("2026-05-01T10:01:00Z"),
    )
    first.close()

    with pytest.raises(QueueFullError):
        Engram(user_id="alice", path=str(tmp_path), queue_max_size=1, queue_put_timeout=0.001)

    retry = Engram(user_id="alice", path=str(tmp_path), queue_max_size=4, queue_put_timeout=0.001)
    assert retry.queue.qsize() == 2
    retry.close()
