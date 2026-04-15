from __future__ import annotations

import gzip
import json

import pytest

import engram.retrieval as retrieval_module
from engram import Engram, ValidationError
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class SequenceExtractor:
    def __init__(self, versions_to_events: list[list[ExtractedEvent]], *, version: str = "reprocess-v1"):
        self.version = version
        self._versions_to_events = [
            [
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
                for event in batch
            ]
            for batch in versions_to_events
        ]
        self.calls: list[str] = []

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        self.calls.append(item.turn_id)
        if not self._versions_to_events:
            raise RuntimeError("no more batches configured")
        return self._versions_to_events.pop(0)


class FailingExtractor:
    version = "reprocess-v1"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        raise RuntimeError("reprocess boom")


def test_reprocess_single_turn_supersedes_prior_successful_run_and_updates_visibility(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
        [
            [
                    ExtractedEvent(
                        type="entity.create",
                        data={"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
                        effective_at_start=dt("2026-05-01T10:00:00Z"),
                        source_role="user",
                        reason="initial extract",
                        time_confidence="exact",
                )
            ],
            [
                    ExtractedEvent(
                        type="entity.create",
                        data={"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
                        effective_at_start=dt("2026-05-03T09:00:00Z"),
                        source_role="user",
                        reason="reprocessed extract",
                        time_confidence="exact",
                )
            ],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn(
        user="나는 부산으로 이사했어",
        assistant="알겠어, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first_processed_at = dt("2026-05-01T10:00:05Z")
    second_processed_at = dt("2026-05-03T09:00:00Z")
    monkeypatch.setattr("engram.canonical.utcnow", lambda: first_processed_at)
    mem.flush("canonical")

    before = mem.get_known_at("user:alice", dt("2026-05-02T00:00:00Z"))
    assert before is not None
    assert before.attrs == {"location": "Seoul"}

    monkeypatch.setattr("engram.canonical.utcnow", lambda: second_processed_at)
    count = mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    assert count == 1

    runs = mem.store.list_extraction_runs()
    assert len(runs) == 2
    assert runs[0].status == "SUCCEEDED"
    assert runs[0].superseded_at == second_processed_at
    assert runs[1].status == "SUCCEEDED"
    assert runs[1].superseded_at is None

    superseded = mem.store.list_superseded_runs()
    assert len(superseded) == 1
    assert superseded[0]["old_run_id"] == runs[0].id
    assert superseded[0]["new_run_id"] == runs[1].id

    before_reprocess = mem.get_known_at("user:alice", dt("2026-05-02T00:00:00Z"))
    after_reprocess = mem.get_known_at("user:alice", dt("2026-05-04T00:00:00Z"))
    valid_now = mem.get_valid_at("user:alice", dt("2026-05-04T00:00:00Z"))
    assert before_reprocess is not None
    assert before_reprocess.attrs == {"location": "Seoul"}
    assert after_reprocess is not None
    assert after_reprocess.attrs == {"location": "Busan"}
    assert valid_now is not None
    assert valid_now.attrs == {"location": "Busan"}

    mem.close()


def test_reprocess_failed_run_preserves_existing_successful_visibility(tmp_path, monkeypatch):
    first = SequenceExtractor(
        [[ExtractedEvent(type="entity.create", data={"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}}, effective_at_start=dt("2026-05-01T10:00:00Z"))]]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=first)
    ack = mem.turn("나는 채식주의자야", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")

    mem.extractor = FailingExtractor()
    mem.canonical_worker.extractor = mem.extractor

    with pytest.raises(RuntimeError, match="reprocess boom"):
        mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)

    runs = mem.store.list_extraction_runs()
    assert sorted(run.status for run in runs) == ["FAILED", "SUCCEEDED"]
    view = mem.get_valid_at("user:alice", dt("2026-05-02T00:00:00Z"))
    assert view is not None
    assert view.attrs == {"diet": "vegetarian"}

    mem.close()


def test_reprocess_flush_projection_retracts_stale_projection_state(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
        [
            [
                    ExtractedEvent(
                        type="entity.create",
                        data={"id": "user:alice", "type": "user", "attrs": {"location": "Seoul", "diet": "vegetarian"}},
                        effective_at_start=dt("2026-05-01T10:00:00Z"),
                    )
                ],
                [
                    ExtractedEvent(
                        type="entity.create",
                        data={"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
                        effective_at_start=dt("2026-05-02T10:00:00Z"),
                    )
                ],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn("나는 서울에 있고 채식주의자야", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")
    mem.flush("projection")
    assert mem.projector.current_snapshot()["user:alice"].attrs == {"location": "Seoul", "diet": "vegetarian"}

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    mem.flush("projection")

    assert mem.projector.current_snapshot()["user:alice"].attrs == {"diet": "vegetarian"}
    valid = mem.get_valid_at("user:alice", dt("2026-05-03T00:00:00Z"))
    assert valid is not None
    assert valid.attrs == {"diet": "vegetarian"}

    mem.close()


def test_reprocess_with_no_new_events_still_retracts_old_projection_state(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
        [
            [
                    ExtractedEvent(
                        type="entity.create",
                        data={"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
                        effective_at_start=dt("2026-05-01T10:00:00Z"),
                    )
                ],
            [],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn("나는 서울에 있어", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")
    mem.flush("projection")
    assert mem.projector.current_snapshot()["user:alice"].attrs == {"location": "Seoul"}

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    mem.flush("projection")

    assert "user:alice" not in mem.projector.current_snapshot()
    assert mem.get_valid_at("user:alice", dt("2026-05-03T00:00:00Z")) is None

    mem.close()


def test_reprocess_range_semantics_follow_raw_append_order(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
        [
            [ExtractedEvent(type="entity.create", data={"id": "user:one", "type": "user", "attrs": {"name": "one"}}, effective_at_start=dt("2026-05-01T10:00:00Z"))],
            [ExtractedEvent(type="entity.create", data={"id": "user:two", "type": "user", "attrs": {"name": "two"}}, effective_at_start=dt("2026-05-01T10:01:00Z"))],
            [ExtractedEvent(type="entity.create", data={"id": "user:three", "type": "user", "attrs": {"name": "three"}}, effective_at_start=dt("2026-05-01T10:02:00Z"))],
            [ExtractedEvent(type="entity.create", data={"id": "user:two", "type": "user", "attrs": {"name": "two-re"}}, effective_at_start=dt("2026-05-02T10:00:00Z"))],
            [ExtractedEvent(type="entity.create", data={"id": "user:three", "type": "user", "attrs": {"name": "three-re"}}, effective_at_start=dt("2026-05-02T10:00:00Z"))],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack1 = mem.turn("one", "ok", observed_at=dt("2026-05-01T10:00:00Z"))
    ack2 = mem.turn("two", "ok", observed_at=dt("2026-05-01T10:01:00Z"))
    ack3 = mem.turn("three", "ok", observed_at=dt("2026-05-01T10:02:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:10:00Z"))
    mem.flush("canonical")

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    count = mem.reprocess(from_turn_id=ack2.turn_id)

    assert count == 2
    assert extractor.calls[:3] == [ack1.turn_id, ack2.turn_id, ack3.turn_id]
    assert extractor.calls[3:] == [ack2.turn_id, ack3.turn_id]

    with pytest.raises(ValidationError, match="from_turn_id .* after to_turn_id"):
        mem.reprocess(from_turn_id=ack3.turn_id, to_turn_id=ack2.turn_id)

    with pytest.raises(ValidationError, match="turn_id not found"):
        mem.reprocess(from_turn_id="missing")

    mem.close()


def test_reprocess_reads_archived_raw_segments(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
        [
            [ExtractedEvent(type="entity.create", data={"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}}, effective_at_start=dt("2026-05-01T10:00:00Z"))],
            [ExtractedEvent(type="entity.create", data={"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}}, effective_at_start=dt("2026-05-02T10:00:00Z"))],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn("부산으로 이사", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:10:00Z"))
    mem.flush("canonical")

    active_path = mem.raw_log.root / "active-000001.jsonl"
    archived_path = mem.raw_log.archived / "000001.jsonl.gz"
    with active_path.open("rb") as source, gzip.open(archived_path, "wb") as target:
        target.write(source.read())
    active_path.unlink()

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    count = mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)

    assert count == 1
    view = mem.get_valid_at("user:alice", dt("2026-05-03T00:00:00Z"))
    assert view is not None
    assert view.attrs == {"location": "Busan"}

    mem.close()


def test_entity_delete_after_reprocess_ignores_superseded_relation_neighbors_for_dirty_rows(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
        [
            [
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "user:alice", "type": "user", "attrs": {"name": "Alice"}},
                    effective_at_start=dt("2026-05-01T09:00:00Z"),
                ),
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "user:bob", "type": "user", "attrs": {"name": "Bob"}},
                    effective_at_start=dt("2026-05-01T09:00:00Z"),
                ),
                ExtractedEvent(
                    type="relation.create",
                    data={
                        "source": "user:alice",
                        "target": "user:bob",
                        "type": "manager",
                        "attrs": {"scope": "engram"},
                    },
                    effective_at_start=dt("2026-05-01T09:00:00Z"),
                ),
            ],
            [
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "user:alice", "type": "user", "attrs": {"name": "Alice"}},
                    effective_at_start=dt("2026-05-02T09:00:00Z"),
                ),
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "user:bob", "type": "user", "attrs": {"name": "Bob"}},
                    effective_at_start=dt("2026-05-02T09:00:00Z"),
                ),
            ],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    ack = mem.turn("Alice and Bob are manager-linked", "ok", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:10:00Z"))
    mem.flush("canonical")
    mem.flush("projection")

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    mem.flush("projection")

    assert mem.store.related_owner_ids_for_entity("user:alice") == []

    mem.append(
        "entity.delete",
        {"id": "user:alice"},
        observed_at=dt("2026-05-03T10:00:00Z"),
    )

    assert set(mem.store.dirty_owner_ids()) == {"user:alice"}

    mem.close()


def test_startup_catch_up_supersedes_prior_successful_run_when_extractor_version_changes(tmp_path, monkeypatch):
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
                    data={"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
                    effective_at_start=dt("2026-05-02T09:00:00Z"),
                    source_role="user",
                    time_confidence="exact",
                )
            ]

    first = Engram(user_id="alice", path=str(tmp_path), extractor=NoopExtractor())
    ack = first.turn("난 채식주의자야", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:01:00Z"))
    first.flush("canonical")
    first.close()

    second = Engram(user_id="alice", path=str(tmp_path), extractor=RealExtractor())
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T09:00:00Z"))
    second.flush("canonical")

    runs = second.store.list_extraction_runs()
    assert len(runs) == 2
    assert runs[0].superseded_at == dt("2026-05-02T09:00:00Z")
    assert runs[1].superseded_at is None
    assert second.store.successful_source_turn_ids("real-v2") == {ack.turn_id}

    view = second.get_valid_at("user:alice", dt("2026-05-03T00:00:00Z"))
    assert view is not None
    assert view.attrs == {"diet": "vegetarian"}

    second.close()
