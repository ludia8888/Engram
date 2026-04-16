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


def test_concurrent_multi_reader_during_background_writes(tmp_path):
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

    for i in range(20):
        mem.turn(user=f"msg {i}", assistant=f"reply {i}", observed_at=dt("2026-05-01T10:00:00Z"))

    errors: list[tuple[str, Exception]] = []

    def reader_loop(label, fn):
        try:
            for _ in range(50):
                fn()
                time.sleep(0.005)
        except Exception as exc:
            errors.append((label, exc))

    threads = [
        threading.Thread(target=reader_loop, args=("get", lambda: mem.get("user:alice"))),
        threading.Thread(target=reader_loop, args=("search", lambda: mem.search("Busan", k=5))),
        threading.Thread(target=reader_loop, args=("context", lambda: mem.context("Busan", max_tokens=200))),
        threading.Thread(target=reader_loop, args=("history", lambda: mem.known_history("user:alice"))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert not errors, f"reader errors: {errors}"
    assert _wait_for(lambda: extractor.call_count >= 20, timeout=15.0)
    assert mem._background_worker.is_alive
    mem.close()


def test_append_and_background_write_seq_monotonic(tmp_path):
    extractor = CountingExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )

    for i in range(15):
        mem.turn(user=f"msg {i}", assistant=f"reply {i}", observed_at=dt("2026-05-01T10:00:00Z"))

    errors: list[Exception] = []

    def append_loop():
        try:
            for i in range(20):
                mem.append(
                    "entity.create",
                    {"id": f"item:{i}", "type": "item", "attrs": {"n": i}},
                    observed_at=dt("2026-05-01T10:00:00Z"),
                )
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=append_loop)
    t.start()
    t.join(timeout=15.0)

    assert not errors, f"append errors: {errors}"
    assert _wait_for(lambda: extractor.call_count >= 15, timeout=15.0)

    seqs = [row[0] for row in mem.conn.execute("SELECT seq FROM events ORDER BY seq").fetchall()]
    assert seqs == list(range(1, len(seqs) + 1))
    mem.close()


def test_reader_connection_refuses_writes(tmp_path):
    import sqlite3

    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        auto_flush=True,
    )
    reader = mem.store._reader_conn
    assert reader is not mem.store._writer_conn
    try:
        reader.execute("DELETE FROM events")
        assert False, "reader should refuse writes"
    except sqlite3.OperationalError:
        pass
    mem.close()


def test_projection_state_atomic_consistency(tmp_path):
    extractor = CountingExtractor()
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=extractor,
        auto_flush=True,
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"name": "Alice"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"name": "Bob"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
    )
    mem.append(
        "relation.create",
        {"source": "user:alice", "target": "user:bob", "type": "friend", "attrs": {}},
        observed_at=dt("2026-05-01T09:00:00Z"),
    )
    mem.flush("all")

    for i in range(20):
        mem.turn(user=f"msg {i}", assistant=f"reply {i}", observed_at=dt("2026-05-01T10:00:00Z"))

    errors: list[Exception] = []

    def check_consistency():
        try:
            for _ in range(200):
                state = mem.projector._state
                assert isinstance(state.version, int)
                assert state.entities is not None
                assert state.relations is not None
                time.sleep(0.002)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=check_consistency)
    t.start()
    t.join(timeout=15.0)

    assert not errors, f"consistency errors: {errors}"
    mem.close()
