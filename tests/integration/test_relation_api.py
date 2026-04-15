from __future__ import annotations

import pytest

from engram import Engram, ValidationError
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class StaticExtractor:
    def __init__(
        self,
        *,
        version: str = "relation-test-v1",
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
                source_role=event.source_role,
                confidence=event.confidence,
                reason=event.reason,
                time_confidence=event.time_confidence,
            )
            for event in self._events
        ]


class SequenceExtractor:
    def __init__(self, batches: list[list[ExtractedEvent]], *, version: str = "relation-reprocess-v1"):
        self.version = version
        self._batches = [
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
            for batch in batches
        ]

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        if not self._batches:
            raise RuntimeError("no more relation batches configured")
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


def test_flush_canonical_persists_relation_event_and_source_target_entities(tmp_path):
    extractor = StaticExtractor(
        events=[
            ExtractedEvent(
                type="relation.create",
                data={
                    "source": "user:alice",
                    "target": "user:bob",
                    "type": "manager",
                    "attrs": {"scope": "engram"},
                },
                effective_at_start=dt("2026-05-01T00:00:00Z"),
                source_role="user",
                confidence=0.93,
                reason="user said Bob is their manager",
                time_confidence="exact",
            )
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    _append_people(mem)
    mem.turn("Bob is my manager", "Okay, I will remember that.", observed_at=dt("2026-05-01T10:00:00Z"))

    mem.flush("canonical")

    assert mem.store.count_extraction_runs() == 1
    assert mem.store.count_events() == 3
    future = dt("2030-01-01T00:00:00Z").isoformat().replace("+00:00", "Z")
    alice_events = mem.store.entity_events_known_visible_at("user:alice", future)
    bob_events = mem.store.entity_events_known_visible_at("user:bob", future)
    relation_event = next(event for event in alice_events if event.type == "relation.create")

    assert relation_event.id in {event.id for event in bob_events}
    event_entities = mem.store.event_entity_ids_for_events([relation_event.id])
    assert set(event_entities[relation_event.id]) == {"user:alice", "user:bob"}

    mem.close()


def test_search_relation_event_returns_source_and_target_entities(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"team": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        reason="relationship was entered manually",
        time_confidence="exact",
    )

    results = mem.search("manager engram", k=5)

    assert len(results) >= 2
    assert {result.entity_id for result in results[:2]} == {"user:alice", "user:bob"}
    assert all("entity" in result.matched_axes for result in results[:2])

    mem.close()


def test_context_known_renders_relation_changes_and_current_relation_summary(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        reason="Bob started managing Alice",
        time_confidence="exact",
    )
    mem.append(
        "relation.update",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"seniority": "staff"},
        },
        observed_at=dt("2026-05-02T10:00:00Z"),
        effective_at_start=dt("2026-05-02T00:00:00Z"),
        source_role="manual",
        reason="manager relationship was updated",
        time_confidence="exact",
    )

    text = mem.context("manager", max_tokens=800)

    assert "## Current State" in text
    assert "manager -> user:bob" in text
    assert "manager <- user:alice" in text
    assert "scope': 'engram'" in text
    assert "seniority': 'staff'" in text
    assert "## Relevant Changes" in text
    assert "relation user:alice -[manager]-> user:bob" in text

    mem.close()


def test_get_relations_known_returns_outgoing_and_incoming_edges(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    alice_relations = mem.get_relations("user:alice")
    bob_relations = mem.get_relations("user:bob")

    assert len(alice_relations) == 1
    assert alice_relations[0].direction == "outgoing"
    assert alice_relations[0].other_entity_id == "user:bob"
    assert len(bob_relations) == 1
    assert bob_relations[0].direction == "incoming"
    assert bob_relations[0].other_entity_id == "user:alice"

    mem.close()


def test_get_relations_known_at_and_known_time_window_validation(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    monkeypatch.setattr("engram.engram.utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    monkeypatch.setattr("engram.engram.utcnow", lambda: dt("2026-05-03T10:00:05Z"))
    mem.append(
        "relation.delete",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
        },
        observed_at=dt("2026-05-03T10:00:00Z"),
        effective_at_start=dt("2026-05-03T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    before = mem.get_relations("user:alice", time_mode="known", at=dt("2026-05-02T00:00:00Z"))
    after = mem.get_relations("user:alice", time_mode="known", at=dt("2026-05-04T00:00:00Z"))

    assert before and before[0].relation_type == "manager"
    assert after == []

    with pytest.raises(ValidationError, match="time_window is not supported"):
        mem.get_relations(
            "user:alice",
            time_mode="known",
            time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-02T00:00:00Z")),
        )

    mem.close()


def test_context_valid_only_shows_relations_active_in_requested_window(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "relation.delete",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
        },
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-10T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    before = mem.context(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-02T00:00:00Z"), dt("2026-05-12T00:00:00Z")),
        max_tokens=800,
    )
    after = mem.context(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-11T00:00:00Z"), dt("2026-05-12T00:00:00Z")),
        max_tokens=800,
    )

    assert "relations_active_in_window" in before
    assert "attrs_as_of_window_end" in before
    assert "manager -> user:bob" in before
    assert "manager -> user:bob" not in after

    mem.close()


def test_get_relations_valid_point_and_window_semantics(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "mentor",
            "attrs": {"cadence": "weekly"},
        },
        observed_at=dt("2026-05-05T10:00:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "relation.delete",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "mentor",
        },
        observed_at=dt("2026-05-06T10:00:00Z"),
        effective_at_start=dt("2026-05-06T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    point_before = mem.get_relations("user:alice", time_mode="valid", at=dt("2026-05-05T12:00:00Z"))
    point_after = mem.get_relations("user:alice", time_mode="valid", at=dt("2026-05-06T12:00:00Z"))
    window = mem.get_relations(
        "user:alice",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
    )

    assert point_before and point_before[0].relation_type == "mentor"
    assert point_after == []
    assert window and window[0].relation_type == "mentor"

    with pytest.raises(ValidationError, match="either at or time_window"):
        mem.get_relations(
            "user:alice",
            time_mode="valid",
            at=dt("2026-05-05T12:00:00Z"),
            time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        )

    mem.close()


def test_valid_window_shows_relation_active_only_in_middle_of_window(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "mentor",
            "attrs": {"cadence": "weekly"},
        },
        observed_at=dt("2026-05-05T10:00:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "relation.delete",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "mentor",
        },
        observed_at=dt("2026-05-06T10:00:00Z"),
        effective_at_start=dt("2026-05-06T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    results = mem.search(
        "mentor weekly",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        k=5,
    )
    text = mem.context(
        "mentor",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        max_tokens=800,
    )

    assert results
    assert {result.entity_id for result in results[:2]} == {"user:alice", "user:bob"}
    assert "mentor -> user:bob" in text
    assert "relations_active_in_window" in text

    mem.close()


def test_valid_point_query_remains_point_in_time_for_relations(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "relation.delete",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
        },
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-10T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    monkeypatch.setattr("engram.retrieval.utcnow", lambda: dt("2026-05-03T00:00:00Z"))
    monkeypatch.setattr("engram.engram.utcnow", lambda: dt("2026-05-03T00:00:00Z"))
    assert mem.search("manager", time_mode="valid", k=5)
    assert "manager -> user:bob" in mem.context("manager", time_mode="valid", max_tokens=600)

    monkeypatch.setattr("engram.retrieval.utcnow", lambda: dt("2026-05-11T00:00:00Z"))
    monkeypatch.setattr("engram.engram.utcnow", lambda: dt("2026-05-11T00:00:00Z"))
    assert mem.search("manager", time_mode="valid", k=5) == []
    assert "manager -> user:bob" not in mem.context("manager", time_mode="valid", max_tokens=600)

    mem.close()


def test_deleted_endpoint_retracts_relation_from_search_context_and_projection(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.flush("projection")

    assert mem.search("manager", k=5)
    assert "manager -> user:bob" in mem.context("manager", max_tokens=600)
    assert "user:alice" in mem.projector.current_relation_snapshot()

    mem.append(
        "entity.delete",
        {"id": "user:bob"},
        observed_at=dt("2026-05-03T10:00:00Z"),
        effective_at_start=dt("2026-05-03T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.flush("projection")

    assert mem.search("manager", k=5) == []
    context = mem.context("manager", max_tokens=600)
    assert "manager -> user:bob" not in context
    assert "user:bob (unknown)" not in context
    assert "user:alice" not in mem.projector.current_relation_snapshot()
    assert "user:bob" not in mem.projector.current_relation_snapshot()

    history = mem.relation_history("user:alice")
    assert len(history) == 1
    assert history[0].action == "create"
    assert history[0].other_entity_id == "user:bob"

    mem.close()


def test_relation_update_without_prior_create_is_treated_as_active_relation(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.update",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "mentor",
            "attrs": {"cadence": "weekly"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        reason="relation update is treated as upsert",
        time_confidence="exact",
    )
    mem.flush("projection")

    assert mem.search("mentor weekly", k=5)
    text = mem.context("mentor", max_tokens=600)
    assert "mentor -> user:bob" in text
    assert "cadence': 'weekly'" in text
    assert mem.projector.current_relation_snapshot()["user:alice"][0].relation_type == "mentor"
    relations = mem.get_relations("user:alice")
    history = mem.relation_history("user:alice")
    assert relations and relations[0].relation_type == "mentor"
    assert len(history) == 1
    assert history[0].action == "update"

    mem.close()


def test_relation_update_without_prior_create_is_visible_in_valid_point_and_window(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.update",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "teammate",
            "attrs": {"project": "engram"},
        },
        observed_at=dt("2026-05-05T10:00:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        source_role="manual",
        reason="relation update is treated as upsert",
        time_confidence="exact",
    )

    window_results = mem.search(
        "teammate engram",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        k=5,
    )
    window_context = mem.context(
        "teammate",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        max_tokens=800,
    )

    monkeypatch.setattr("engram.retrieval.utcnow", lambda: dt("2026-05-06T00:00:00Z"))
    monkeypatch.setattr("engram.engram.utcnow", lambda: dt("2026-05-06T00:00:00Z"))
    point_results = mem.search("teammate engram", time_mode="valid", k=5)
    point_context = mem.context("teammate", time_mode="valid", max_tokens=800)

    assert window_results
    assert point_results
    assert "teammate -> user:bob" in window_context
    assert "relations_active_in_window" in window_context
    assert "teammate -> user:bob" in point_context

    mem.close()


def test_valid_window_only_shows_relation_when_endpoint_overlaps_window(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "entity.delete",
        {"id": "user:bob"},
        observed_at=dt("2026-05-05T10:00:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    visible_results = mem.search(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-02T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        k=5,
    )
    visible_context = mem.context(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-02T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        max_tokens=800,
    )
    hidden_results = mem.search(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-06T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        k=5,
    )
    hidden_context = mem.context(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-06T00:00:00Z"), dt("2026-05-10T00:00:00Z")),
        max_tokens=800,
    )

    assert visible_results
    assert "manager -> user:bob" in visible_context
    assert hidden_results == []
    assert "manager -> user:bob" not in hidden_context

    mem.close()


def test_reprocess_retracts_stale_relation_from_search_context_and_projection(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
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
                    effective_at_start=dt("2026-05-01T00:00:00Z"),
                    source_role="user",
                    time_confidence="exact",
                )
            ],
            [],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    _append_people(mem)
    ack = mem.turn("Bob is my manager", "Okay", observed_at=dt("2026-05-01T10:00:00Z"))

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-04-01T10:00:05Z"))
    mem.flush("canonical")
    mem.flush("projection")

    assert mem.projector.current_relation_snapshot()["user:alice"][0].relation_type == "manager"
    assert mem.search("manager", k=5)

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-04-02T10:00:00Z"))
    count = mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    assert count == 1
    mem.flush("projection")

    assert "user:alice" not in mem.projector.current_relation_snapshot()
    assert "user:bob" not in mem.projector.current_relation_snapshot()
    assert mem.search("manager", k=5) == []
    assert (
        "manager -> user:bob"
        not in mem.context(
            "manager",
            max_tokens=400,
        )
    )

    mem.close()


def test_relation_history_returns_create_update_delete_and_supports_filters(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    _append_people(mem)
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "relation.update",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"seniority": "staff"},
        },
        observed_at=dt("2026-05-02T10:00:00Z"),
        effective_at_start=dt("2026-05-02T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )
    mem.append(
        "relation.delete",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
        },
        observed_at=dt("2026-05-03T10:00:00Z"),
        effective_at_start=dt("2026-05-03T00:00:00Z"),
        source_role="manual",
        time_confidence="exact",
    )

    history = mem.relation_history("user:alice")
    filtered = mem.relation_history("user:alice", relation_type="manager", other_entity_id="user:bob")

    assert [entry.action for entry in history] == ["create", "update", "delete"]
    assert all(entry.direction == "outgoing" for entry in history)
    assert len(filtered) == 3

    mem.close()


def test_relation_history_hides_superseded_runs_for_known_and_valid(tmp_path, monkeypatch):
    extractor = SequenceExtractor(
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
                    effective_at_start=dt("2026-05-01T00:00:00Z"),
                    source_role="user",
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
                    effective_at_start=dt("2026-05-02T00:00:00Z"),
                    source_role="user",
                    time_confidence="exact",
                )
            ],
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    _append_people(mem)
    ack = mem.turn("Bob role changed", "Okay", observed_at=dt("2026-05-01T10:00:00Z"))

    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-01T10:00:05Z"))
    mem.flush("canonical")
    monkeypatch.setattr("engram.canonical.utcnow", lambda: dt("2026-05-02T10:00:00Z"))
    mem.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    monkeypatch.setattr("engram.engram.utcnow", lambda: dt("2026-05-03T00:00:00Z"))

    known_history = mem.relation_history("user:alice", time_mode="known")
    valid_history = mem.relation_history("user:alice", time_mode="valid")

    assert [entry.relation_type for entry in known_history] == ["mentor"]
    assert [entry.relation_type for entry in valid_history] == ["mentor"]

    mem.close()
