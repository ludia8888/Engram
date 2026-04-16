from __future__ import annotations

import pytest

from engram import Engram, QueueFullError
from engram.types import ExtractedEvent

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
    class StaticExtractor:
        version = "startup-test-v1"

        def extract(self, item):
            return [
                ExtractedEvent(
                    type="entity.create",
                    data={
                        "id": "user:alice",
                        "type": "user",
                        "attrs": {"diet": "vegetarian"},
                    },
                    source_role="user",
                    time_confidence="exact",
                )
            ]

    first = Engram(user_id="alice", path=str(tmp_path), extractor=StaticExtractor())
    first.turn(
        user="난 채식주의자야",
        assistant="알겠어, 식단 선호로 기억할게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("canonical")
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path), extractor=StaticExtractor())

    assert second.queue.qsize() == 0

    second.close()


def test_startup_catch_up_requeues_turn_when_extractor_version_changes(tmp_path):
    class NoopExtractor:
        version = "noop-v1"

        def extract(self, item):
            return []

    class RealExtractor:
        version = "real-v2"

        def extract(self, item):
            return [
                ExtractedEvent(
                    type="entity.create",
                    data={
                        "id": "user:alice",
                        "type": "user",
                        "attrs": {"diet": "vegetarian"},
                    },
                    source_role="user",
                    time_confidence="exact",
                )
            ]

    first = Engram(user_id="alice", path=str(tmp_path), extractor=NoopExtractor())
    ack = first.turn(
        user="난 채식주의자야",
        assistant="알겠어, 식단 선호로 기억할게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("canonical")
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path), extractor=RealExtractor())

    assert second.queue.qsize() == 1
    queued = second.queue.get_nowait()
    assert queued.turn_id == ack.turn_id

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


def test_startup_loads_snapshot_then_rebuilds_delta(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("all")

    snapshot = first.projector.current_snapshot()
    assert "user:alice" in snapshot

    first.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
    )
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    snapshot = second.projector.current_snapshot()
    assert "user:alice" in snapshot
    assert snapshot["user:alice"].attrs == {"diet": "vegetarian", "location": "Busan"}
    assert second.store.count_dirty_ranges() == 0

    second.close()


def test_corrupt_snapshot_does_not_block_startup(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("all")
    assert first.store.load_latest_snapshot() is not None

    first.conn.execute(
        "UPDATE snapshots SET state_blob = X'DEADBEEF', relation_blob = X'DEADBEEF'"
    )
    first.conn.commit()
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    assert second.store.load_latest_snapshot() is None
    view = second.get("user:alice")
    assert view is not None
    assert view.attrs == {"diet": "vegetarian"}

    second.close()


def test_stale_snapshot_with_no_dirty_ranges_still_rebuilds(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("all")

    snapshot = first.store.load_latest_snapshot()
    assert snapshot is not None
    stale_seq = snapshot.last_seq

    first.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
    )
    first.flush("projection")

    assert first.store.count_dirty_ranges() == 0
    assert first.store.current_max_seq() > stale_seq
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    assert second.store.count_dirty_ranges() == 0
    view = second.get("user:alice")
    assert view is not None
    assert view.attrs == {"diet": "vegetarian", "location": "Busan"}

    snapshot_after = second.projector.current_snapshot()
    assert "user:alice" in snapshot_after
    assert snapshot_after["user:alice"].attrs == {"diet": "vegetarian", "location": "Busan"}

    second.close()


def test_startup_recovery_backfills_missing_semantic_index_rows(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path))
    first.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="moved to Busan",
    )
    assert first.store.count_vec_events(first.embedder.version) == 0
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path))

    assert second.store.count_vec_events(second.embedder.version) >= 1

    second.close()
