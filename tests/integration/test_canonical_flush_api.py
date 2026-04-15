from __future__ import annotations

import pytest

from engram import Engram
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class StaticExtractor:
    def __init__(
        self,
        *,
        version: str = "test-extractor-v1",
        events: list[ExtractedEvent] | None = None,
        error: Exception | None = None,
    ):
        self.version = version
        self._events = list(events or [])
        self._error = error

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        if self._error is not None:
            raise self._error
        return [
            ExtractedEvent(
                type=event.type,
                data=dict(event.data),
                effective_at_start=event.effective_at_start,
                effective_at_end=event.effective_at_end,
                source_role=event.source_role,
                confidence=event.confidence,
                reason=event.reason,
                time_confidence=event.time_confidence,
            )
            for event in self._events
        ]


def test_flush_canonical_drains_queue_records_run_and_appends_events(tmp_path):
    extractor = StaticExtractor(
        events=[
            ExtractedEvent(
                type="entity.create",
                data={
                    "id": "user:alice",
                    "type": "user",
                    "attrs": {"diet": "vegetarian"},
                },
                source_role="user",
                confidence=0.92,
                reason="user explicitly stated their diet",
                time_confidence="exact",
            )
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 식단 선호로 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    assert mem.queue.qsize() == 1

    mem.flush("canonical")

    assert mem.queue.qsize() == 0
    assert mem.store.count_extraction_runs() == 1
    assert mem.store.count_events() == 1
    assert mem.store.successful_source_turn_ids() == {ack.turn_id}
    assert mem.get("user:alice").attrs == {"diet": "vegetarian"}

    run = mem.store.list_extraction_runs()[0]
    assert run.status == "SUCCEEDED"
    assert run.source_turn_id == ack.turn_id
    assert run.extractor_version == "test-extractor-v1"
    assert run.event_count == 1

    mem.close()


def test_flush_canonical_records_success_even_when_extractor_emits_no_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=StaticExtractor(events=[]))
    ack = mem.turn(
        user="안녕",
        assistant="안녕!",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    mem.flush("canonical")

    assert mem.queue.qsize() == 0
    assert mem.store.count_extraction_runs() == 1
    assert mem.store.count_events() == 0
    assert mem.store.successful_source_turn_ids() == {ack.turn_id}

    mem.close()


def test_flush_canonical_skips_duplicate_turn_when_successful_run_exists(tmp_path):
    extractor = StaticExtractor(
        events=[
            ExtractedEvent(
                type="entity.create",
                data={
                    "id": "user:alice",
                    "type": "user",
                    "attrs": {"diet": "vegetarian"},
                },
            )
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    mem.flush("canonical")
    mem.queue.put_nowait(QueueItem.from_turn(mem.raw_get(ack.turn_id)))
    mem.flush("canonical")

    assert mem.store.count_extraction_runs() == 1
    assert mem.store.count_events() == 1

    mem.close()


def test_flush_canonical_records_failed_run_and_raises(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(error=RuntimeError("extract boom")),
    )
    mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    with pytest.raises(RuntimeError, match="extract boom"):
        mem.flush("canonical")

    assert mem.queue.qsize() == 0
    assert mem.store.count_extraction_runs() == 1
    assert mem.store.count_events() == 0
    run = mem.store.list_extraction_runs()[0]
    assert run.status == "FAILED"
    assert run.error == "extract boom"

    mem.close()


def test_startup_catch_up_skips_turns_with_successful_empty_run(tmp_path):
    first = Engram(user_id="alice", path=str(tmp_path), extractor=StaticExtractor(events=[]))
    first.turn(
        user="안녕",
        assistant="안녕!",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("canonical")
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path), extractor=StaticExtractor(events=[]))

    assert second.queue.qsize() == 0
    assert second.store.count_extraction_runs() == 1

    second.close()
