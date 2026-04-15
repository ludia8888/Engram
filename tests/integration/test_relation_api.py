from __future__ import annotations

from engram import Engram
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
        time_window=(dt("2026-05-02T00:00:00Z"), dt("2026-05-03T00:00:00Z")),
        max_tokens=800,
    )
    after = mem.context(
        "manager",
        time_mode="valid",
        time_window=(dt("2026-05-11T00:00:00Z"), dt("2026-05-12T00:00:00Z")),
        max_tokens=800,
    )

    assert "manager -> user:bob" in before
    assert "manager -> user:bob" not in after

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
