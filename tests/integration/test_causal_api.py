from __future__ import annotations

import pytest

import engram.canonical as canonical_module
import engram.engram as engram_module
import engram.retrieval as retrieval_module
from engram import Engram, ValidationError
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class StaticExtractor:
    def __init__(
        self,
        *,
        version: str = "causal-test-v1",
        events: list[ExtractedEvent] | None = None,
    ):
        self.version = version
        self._events = list(events or [])

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        return [
            ExtractedEvent(
                type=event.type,
                data=dict(event.data),
                effective_at_start=event.effective_at_start,
                effective_at_end=event.effective_at_end,
                caused_by=event.caused_by,
                source_role=event.source_role,
                confidence=event.confidence,
                reason=event.reason,
                time_confidence=event.time_confidence,
            )
            for event in self._events
        ]


class SequenceExtractor:
    def __init__(self, batches: list[list[ExtractedEvent]], *, version: str = "causal-seq-v1"):
        self.version = version
        self._batches = [
            [
                ExtractedEvent(
                    type=event.type,
                    data=dict(event.data),
                    effective_at_start=event.effective_at_start,
                    effective_at_end=event.effective_at_end,
                    caused_by=event.caused_by,
                    source_role=event.source_role,
                    confidence=event.confidence,
                    reason=event.reason,
                    time_confidence=event.time_confidence,
                )
                for event in batch
            ]
            for batch in batches
        ]

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        if not self._batches:
            raise RuntimeError("no more causal batches configured")
        return self._batches.pop(0)


def _append_people(mem: Engram) -> None:
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"name": "Alice"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"name": "Bob"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )


def test_append_accepts_existing_caused_by_and_rejects_missing_event(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    cause_id = mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    effect_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"status": "promoted"}},
        observed_at=dt("2026-05-02T10:00:00Z"),
        effective_at_start=dt("2026-05-02T00:00:00Z"),
        caused_by=cause_id,
        source_role="manual",
        time_confidence="exact",
    )

    effect = mem.store.event_by_id(effect_id)
    assert effect is not None
    assert effect.caused_by == cause_id

    with pytest.raises(ValidationError, match="caused_by event not found"):
        mem.append(
            "entity.update",
            {"id": "user:alice", "attrs": {"status": "missing-cause"}},
            observed_at=dt("2026-05-03T10:00:00Z"),
            effective_at_start=dt("2026-05-03T00:00:00Z"),
            caused_by="missing-event-id",
            source_role="manual",
            time_confidence="exact",
        )

    mem.close()


def test_flush_canonical_persists_valid_caused_by_link(tmp_path, monkeypatch):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(),
    )
    cause_id = mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"topic": "promotion"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.extractor = StaticExtractor(
        events=[
            ExtractedEvent(
                type="entity.update",
                data={"id": "user:alice", "attrs": {"role": "manager"}},
                effective_at_start=dt("2026-05-01T10:00:00Z"),
                caused_by=cause_id,
                source_role="user",
                reason="promotion caused new role",
                time_confidence="exact",
            )
        ]
    )
    mem.canonical_worker.extractor = mem.extractor
    mem.turn("승진해서 매니저가 됐어", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))

    monkeypatch.setattr(canonical_module, "utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")

    caused = [
        event
        for event in mem.store.visible_events_known(dt("2026-05-02T00:00:00Z").isoformat().replace("+00:00", "Z"))
        if event.caused_by is not None
    ]
    assert len(caused) == 1
    assert caused[0].caused_by == cause_id

    mem.close()


def test_reprocess_invalid_caused_by_records_failed_run_and_preserves_active_run(tmp_path, monkeypatch):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(
            events=[
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "user:alice", "type": "user", "attrs": {"role": "manager"}},
                    effective_at_start=dt("2026-05-01T10:00:00Z"),
                    source_role="user",
                    time_confidence="exact",
                )
            ]
        ),
    )
    ack = mem.turn("매니저가 됐어", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))

    monkeypatch.setattr(canonical_module, "utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")

    mem.extractor = StaticExtractor(
        version="causal-test-v2",
        events=[
            ExtractedEvent(
                type="entity.create",
                data={"id": "user:alice", "type": "user", "attrs": {"role": "director"}},
                effective_at_start=dt("2026-05-02T10:00:00Z"),
                caused_by="missing-event-id",
                source_role="user",
                time_confidence="exact",
            )
        ],
    )
    mem.canonical_worker.extractor = mem.extractor

    with pytest.raises(ValidationError, match="caused_by event not found"):
        mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)

    statuses = sorted(run.status for run in mem.store.list_extraction_runs())
    assert statuses == ["FAILED", "SUCCEEDED"]
    view = mem.get_valid_at("user:alice", dt("2026-05-03T00:00:00Z"))
    assert view is not None
    assert view.attrs == {"role": "manager"}

    mem.close()


def test_search_known_and_context_expand_explicit_causal_neighbors(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    cause_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        reason="Alice moved to Busan",
        time_confidence="exact",
    )
    effect_id = mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-02T10:00:00Z"),
        effective_at_start=dt("2026-05-02T00:00:00Z"),
        caused_by=cause_id,
        source_role="manual",
        reason="move triggered manager reassignment",
        time_confidence="exact",
    )

    results = mem.search("Busan", k=5)
    bob = next(result for result in results if result.entity_id == "user:bob")

    assert "causal" in bob.matched_axes
    assert effect_id in bob.supporting_event_ids

    text = mem.context("manager", max_tokens=900)
    assert "causal_support: yes" in text
    assert "caused by: user:alice {'location': 'Busan'}" in text
    assert "manager -> user:bob" in text

    mem.close()


def test_causal_scoring_does_not_boost_events_that_are_already_direct_hits(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"name": "Alice"}},
        observed_at=dt("2026-05-01T09:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    cause_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"topic": "alpha"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T10:00:00Z"),
        source_role="manual",
        reason="alpha topic",
        time_confidence="exact",
    )
    effect_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"note": "beta"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
        effective_at_start=dt("2026-05-01T11:00:00Z"),
        caused_by=cause_id,
        source_role="manual",
        reason="beta note",
        time_confidence="exact",
    )

    results = mem.search("alpha beta", k=5)

    assert len(results) == 1
    assert results[0].entity_id == "user:alice"
    assert results[0].score == 1.0
    assert "causal" not in results[0].matched_axes
    assert results[0].supporting_event_ids == [cause_id, effect_id]

    mem.close()


def test_known_history_and_valid_context_respect_visible_causal_run_after_reprocess(tmp_path, monkeypatch):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=SequenceExtractor([]),
    )
    _append_people(mem)
    cause_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"topic": "promotion"}},
        observed_at=dt("2026-05-01T09:30:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    mem.extractor = SequenceExtractor(
        [
            [
                ExtractedEvent(
                    type="relation.create",
                    data={
                        "source": "user:alice",
                        "target": "user:bob",
                        "type": "manager",
                        "attrs": {"scope": "engram"},
                    },
                    effective_at_start=dt("2026-05-01T10:00:00Z"),
                    caused_by=cause_id,
                    source_role="user",
                    reason="promotion led to manager assignment",
                    time_confidence="exact",
                )
            ],
            [
                ExtractedEvent(
                    type="relation.create",
                    data={
                        "source": "user:alice",
                        "target": "user:bob",
                        "type": "mentor",
                        "attrs": {"scope": "engram"},
                    },
                    effective_at_start=dt("2026-05-02T10:00:00Z"),
                    caused_by=cause_id,
                    source_role="user",
                    reason="promotion was reinterpreted as mentorship",
                    time_confidence="exact",
                )
            ],
        ],
        version="causal-seq-v1",
    )
    mem.canonical_worker.extractor = mem.extractor

    ack = mem.turn("승진했고 Bob과 역할 관계가 바뀌었어", "알겠어", observed_at=dt("2026-05-01T10:00:00Z"))
    monkeypatch.setattr(canonical_module, "utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")

    old_context = mem.context(
        "promotion",
        time_mode="known",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-01T12:00:00Z")),
        max_tokens=900,
    )
    assert "manager -> user:bob" in old_context
    assert "mentor -> user:bob" not in old_context

    monkeypatch.setattr(canonical_module, "utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    monkeypatch.setattr(retrieval_module, "utcnow", lambda: dt("2026-05-03T00:00:00Z"))
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-03T00:00:00Z"))

    valid_context = mem.context(
        "promotion",
        time_mode="valid",
        max_tokens=900,
    )
    assert "mentor -> user:bob" in valid_context
    assert "manager -> user:bob" not in valid_context
    assert "caused by: user:alice {'topic': 'promotion'}" in valid_context

    mem.close()
