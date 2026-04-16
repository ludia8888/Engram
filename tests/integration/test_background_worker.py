from __future__ import annotations

import time

from engram import Engram
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class SimpleExtractor:
    version = "simple-v1"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        return [
            ExtractedEvent(
                type="entity.create",
                data={"id": "user:alice", "type": "user", "attrs": {"seen": True}},
                source_role="user",
                confidence=0.9,
                reason="test",
            )
        ]


class FailOnceExtractor:
    version = "fail-once-v1"

    def __init__(self):
        self._calls = 0

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("transient")
        return [
            ExtractedEvent(
                type="entity.create",
                data={"id": "user:alice", "type": "user", "attrs": {"recovered": True}},
                source_role="user",
                confidence=0.9,
                reason="recovered",
            )
        ]


class MutableExtractor:
    version = "mutable-v1"

    def __init__(self):
        self.location = "Seoul"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        return [
            ExtractedEvent(
                type="entity.create",
                data={"id": "user:alice", "type": "user", "attrs": {"location": self.location}},
                source_role="user",
                confidence=0.9,
                reason=f"location is {self.location}",
            )
        ]


def _wait_for(pred, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_auto_flush_processes_turn_automatically(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=SimpleExtractor(),
        auto_flush=True,
    )
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))

    assert _wait_for(lambda: mem.store.count_events() > 0)
    assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0)
    assert mem.get("user:alice") is not None

    mem.close()


def test_auto_flush_does_not_crash_on_failure(tmp_path):
    class AlwaysFail:
        version = "fail-v1"

        def extract(self, item):
            raise RuntimeError("boom")

    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=AlwaysFail(),
        auto_flush=True,
    )
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))
    time.sleep(0.5)

    assert mem._background_worker.is_alive
    mem.close()


def test_auto_flush_retries_failed_turn(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=FailOnceExtractor(),
        auto_flush=True,
    )
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))

    assert _wait_for(lambda: mem.get("user:alice") is not None, timeout=10.0)
    assert mem.get("user:alice").attrs.get("recovered") is True

    mem.close()


def test_auto_flush_clean_shutdown(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=SimpleExtractor(),
        auto_flush=True,
    )
    worker = mem._background_worker
    assert worker.is_alive
    mem.close()
    assert not worker.is_alive


def test_auto_flush_is_opt_in(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    assert mem._background_worker is None
    mem.close()


def test_auto_flush_processes_multiple_turns(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=SimpleExtractor(),
        auto_flush=True,
    )
    for i in range(3):
        mem.turn(user=f"msg {i}", assistant=f"reply {i}", observed_at=dt("2026-05-01T10:00:00Z"))

    assert _wait_for(lambda: mem.store.count_extraction_runs() >= 3, timeout=5.0)

    mem.close()


def test_auto_flush_persists_snapshot(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=SimpleExtractor(),
        auto_flush=True,
    )
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))

    assert _wait_for(lambda: mem.store.load_latest_snapshot() is not None, timeout=5.0)

    mem.close()


def test_auto_flush_append_triggers_projection_and_index(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=True,
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=5.0)
    assert _wait_for(lambda: mem.store.load_latest_snapshot() is not None, timeout=5.0)
    assert _wait_for(lambda: mem.store.count_vec_events(mem.embedder.version) >= 1, timeout=5.0)
    assert mem.get("user:alice") is not None

    mem.close()


def test_auto_flush_append_multiple_requests_are_not_lost(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=True,
    )

    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"diet": "vegan"}},
        observed_at=dt("2026-05-01T10:00:01Z"),
    )

    assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=5.0)
    assert _wait_for(lambda: mem.store.count_vec_events(mem.embedder.version) >= 2, timeout=5.0)
    entity = mem.get("user:alice")
    assert entity is not None
    assert entity.attrs["location"] == "Busan"
    assert entity.attrs["diet"] == "vegan"

    mem.close()


def test_append_auto_flush_is_opt_in(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=False,
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    time.sleep(0.2)

    assert mem.store.count_dirty_ranges() > 0
    assert mem.store.load_latest_snapshot() is None
    assert mem.store.count_vec_events(mem.embedder.version) == 0

    mem.close()


def test_flush_canonical_with_auto_flush_restarts_worker_and_requests_maintenance(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=SimpleExtractor(),
        auto_flush=True,
    )
    worker = mem._background_worker
    worker.stop()

    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))
    assert mem.queue.qsize() == 1

    worker.start()
    mem.flush("canonical")

    assert worker.is_alive
    assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=5.0)
    assert _wait_for(lambda: mem.store.load_latest_snapshot() is not None, timeout=5.0)
    assert _wait_for(lambda: mem.store.count_vec_events(mem.embedder.version) >= 1, timeout=5.0)

    mem.close()


def test_reprocess_with_auto_flush_requests_followup_maintenance(tmp_path):
    extractor = MutableExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )
    ack = mem.turn(user="I moved", assistant="ok", observed_at=dt("2026-05-01T10:00:00Z"))

    assert _wait_for(lambda: mem.get("user:alice") is not None, timeout=5.0)
    assert mem.get("user:alice").attrs["location"] == "Seoul"

    extractor.location = "Busan"
    assert mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id) == 1

    assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=5.0)
    assert _wait_for(lambda: mem.get("user:alice").attrs["location"] == "Busan", timeout=5.0)

    mem.close()
