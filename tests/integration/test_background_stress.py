from __future__ import annotations

import threading
import time

from engram import Engram
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class CountingExtractor:
    version = "counting-v1"

    def __init__(self):
        self._lock = threading.Lock()
        self.call_count = 0

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        with self._lock:
            self.call_count += 1
        return [
            ExtractedEvent(
                type="entity.update",
                data={"id": "user:alice", "attrs": {f"turn_{self.call_count}": True}},
                source_role="user",
                confidence=0.9,
                reason=f"turn {self.call_count}",
            )
        ]


def _wait_for(pred, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_rapid_fire_turns_while_worker_processes(tmp_path):
    extractor = CountingExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"init": True}},
        observed_at=dt("2026-05-01T09:00:00Z"),
    )

    n = 20
    for i in range(n):
        mem.turn(
            user=f"message {i}",
            assistant=f"reply {i}",
            observed_at=dt("2026-05-01T10:00:00Z"),
        )

    assert _wait_for(lambda: extractor.call_count >= n, timeout=15.0)
    assert _wait_for(lambda: mem.store.count_dirty_ranges() == 0, timeout=10.0)
    assert mem._background_worker.is_alive

    entity = mem.get("user:alice")
    assert entity is not None

    mem.close()


def test_concurrent_append_and_auto_flush(tmp_path):
    extractor = CountingExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )

    mem.turn(user="trigger", assistant="ok", observed_at=dt("2026-05-01T10:00:00Z"))

    errors: list[Exception] = []

    def append_loop():
        try:
            for i in range(10):
                mem.append(
                    "entity.create",
                    {"id": f"item:{i}", "type": "item", "attrs": {"n": i}},
                    observed_at=dt("2026-05-01T10:00:00Z"),
                )
                time.sleep(0.01)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=append_loop)
    t.start()
    t.join(timeout=10.0)

    assert not errors, f"append raised: {errors}"
    assert _wait_for(lambda: extractor.call_count >= 1, timeout=10.0)
    assert mem._background_worker.is_alive

    mem.close()


def test_manual_flush_concurrent_with_auto_flush(tmp_path):
    extractor = CountingExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )

    for i in range(5):
        mem.turn(user=f"msg {i}", assistant=f"reply {i}", observed_at=dt("2026-05-01T10:00:00Z"))

    errors: list[Exception] = []

    def manual_flush():
        try:
            mem.flush("projection")
            mem.flush("snapshot")
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=manual_flush)
    t.start()
    t.join(timeout=10.0)

    assert not errors, f"manual flush raised: {errors}"
    assert _wait_for(lambda: extractor.call_count >= 5, timeout=10.0)
    assert mem._background_worker.is_alive

    mem.close()


def test_search_during_background_processing(tmp_path):
    extractor = CountingExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
    )
    mem.flush("all")

    for i in range(10):
        mem.turn(user=f"msg {i}", assistant=f"reply {i}", observed_at=dt("2026-05-01T10:00:00Z"))

    errors: list[Exception] = []

    def search_loop():
        try:
            for _ in range(20):
                mem.search("Busan", k=5)
                mem.context("Busan", max_tokens=500)
                time.sleep(0.02)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=search_loop)
    t.start()
    t.join(timeout=15.0)

    assert not errors, f"search raised: {errors}"
    assert mem._background_worker.is_alive

    mem.close()


def test_worker_survives_rapid_open_close_cycle(tmp_path):
    for cycle in range(3):
        mem = Engram(
            user_id="alice",
            path=str(tmp_path),
            extractor=CountingExtractor(),
            auto_flush=True,
        )
        mem.turn(
            user=f"cycle {cycle}",
            assistant="ok",
            observed_at=dt("2026-05-01T10:00:00Z"),
        )
        time.sleep(0.2)
        mem.close()

    final = Engram(user_id="alice", path=str(tmp_path))
    runs = final.store.list_extraction_runs()
    assert len(runs) >= 1
    final.close()
