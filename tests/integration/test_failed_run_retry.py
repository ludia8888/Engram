from __future__ import annotations

from engram import Engram
from engram.retry import RetryPolicy
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class FailOnceExtractor:
    version = "fail-once-v1"

    def __init__(self):
        self._calls = 0

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("transient failure")
        return [
            ExtractedEvent(
                type="entity.create",
                data={"id": "user:alice", "type": "user", "attrs": {"seen": True}},
                source_role="user",
                confidence=0.9,
                reason="test",
            )
        ]


class AlwaysFailExtractor:
    version = "always-fail-v1"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        raise RuntimeError("permanent failure")


def test_process_with_retry_succeeds_after_transient_failure(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=FailOnceExtractor())
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))

    policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=0.0)

    item = mem.queue.get_nowait()
    mem.queue.task_done()
    success1, state1 = mem.canonical_worker.process_with_retry(item, policy)
    assert not success1
    assert state1 is not None
    assert state1.attempt == 1

    success2, state2 = mem.canonical_worker.process_with_retry(item, policy)
    assert success2
    assert state2 is None

    assert mem.get("user:alice") is not None
    mem.close()


def test_process_with_retry_gives_up_after_max_retries(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=AlwaysFailExtractor())
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))

    policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=0.0)

    item = mem.queue.get_nowait()
    mem.queue.task_done()

    success1, state1 = mem.canonical_worker.process_with_retry(item, policy)
    assert not success1
    assert state1 is not None

    success2, state2 = mem.canonical_worker.process_with_retry(item, policy)
    assert not success2
    assert state2 is None

    runs = mem.store.list_extraction_runs()
    assert all(run.status == "FAILED" for run in runs)
    assert len(runs) == 2
    mem.close()


def test_process_with_retry_skips_already_succeeded_turn(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"ok": True}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.turn(user="hello", assistant="hi", observed_at=dt("2026-05-01T10:00:00Z"))
    mem.flush("canonical")

    item = QueueItem(
        turn_id=mem.store.list_extraction_runs()[0].source_turn_id,
        observed_at=dt("2026-05-01T10:00:00Z"),
        session_id=None,
        user="hello",
        assistant="hi",
        metadata={},
    )

    policy = RetryPolicy(max_retries=3)
    success, state = mem.canonical_worker.process_with_retry(item, policy)
    assert not success
    assert state is None
    mem.close()
