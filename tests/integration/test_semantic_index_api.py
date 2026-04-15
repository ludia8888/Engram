from __future__ import annotations

import engram.canonical as canonical_module

from engram import Engram
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class StaticExtractor:
    def __init__(
        self,
        *,
        version: str = "semantic-test-extractor-v1",
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


class StaticEmbedder:
    def __init__(
        self,
        *,
        version: str = "static-embedder-v1",
        dim: int = 3,
        mapping: dict[str, list[float]] | None = None,
    ):
        self.version = version
        self.dim = dim
        self._mapping = {key: list(value) for key, value in (mapping or {}).items()}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._mapping.get(text, [0.0] * self.dim)) for text in texts]


def test_flush_index_backfills_vec_events_for_current_embedder_version(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(
            mapping={
                "entity.create {\"attrs\": {\"diet\": \"vegetarian\"}, \"id\": \"user:alice\", \"type\": \"user\"} user": [1.0, 0.0, 0.0],
            }
        ),
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        source_role="user",
    )

    assert mem.store.count_vec_events("static-embedder-v1") == 0

    mem.flush("index")

    assert mem.store.count_vec_events("static-embedder-v1") == 1
    assert mem.store.missing_event_embedding_ids("static-embedder-v1") == []

    mem.close()


def test_flush_index_keeps_old_versions_and_backfills_new_one(tmp_path):
    first = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(version="embed-v1", mapping={"entity.create {\"attrs\": {\"diet\": \"vegetarian\"}, \"id\": \"user:alice\", \"type\": \"user\"} manual": [1.0, 0.0, 0.0]}),
    )
    first.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("index")
    first.close()

    second = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(version="embed-v2", mapping={"entity.create {\"attrs\": {\"diet\": \"vegetarian\"}, \"id\": \"user:alice\", \"type\": \"user\"} manual": [0.0, 1.0, 0.0]}),
    )

    assert second.store.count_vec_events("embed-v1") == 1
    assert second.store.count_vec_events("embed-v2") == 0

    second.flush("index")

    assert second.store.count_vec_events("embed-v1") == 1
    assert second.store.count_vec_events("embed-v2") == 1

    second.close()


def test_search_can_match_semantic_only_when_lexical_score_is_zero(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"food\": \"noodle-soup\"}, \"id\": \"food:ramen\", \"type\": \"food\"} manual": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"food\": \"salad\"}, \"id\": \"food:salad\", \"type\": \"food\"} manual": [0.0, 1.0, 0.0],
            }
        ),
    )
    mem.append(
        "entity.create",
        {"id": "food:ramen", "type": "food", "attrs": {"food": "noodle-soup"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.append(
        "entity.create",
        {"id": "food:salad", "type": "food", "attrs": {"food": "salad"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
    )
    mem.flush("index")

    results = mem.search("ramen", k=5)

    assert results
    assert results[0].entity_id == "food:ramen"
    assert "semantic" in results[0].matched_axes

    mem.close()


def test_search_uses_lexical_only_when_current_embedder_version_has_no_rows(tmp_path):
    first = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(version="embed-v1", mapping={"Busan": [1.0, 0.0, 0.0]}),
    )
    first.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="moved to Busan",
    )
    first.flush("index")
    first.close()

    second = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(version="embed-v2", mapping={"Busan": [0.0, 1.0, 0.0]}),
    )

    results = second.search("Busan", k=5)

    assert results
    assert results[0].entity_id == "user:alice"
    assert "semantic" not in results[0].matched_axes

    second.close()


def test_search_does_not_penalize_unindexed_lexical_hit_when_other_events_have_semantic_rows(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(
            mapping={
                "ramen busan": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"dish\": \"ramen\"}, \"id\": \"food:old\", \"type\": \"food\"} manual": [1.0, 0.0, 0.0],
            },
        ),
    )
    mem.append(
        "entity.create",
        {"id": "food:old", "type": "food", "attrs": {"dish": "ramen"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.flush("index")
    mem.append(
        "entity.create",
        {"id": "food:new", "type": "food", "attrs": {"dish": "ramen", "location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
    )

    results = mem.search("ramen busan", k=5)

    assert results
    assert results[0].entity_id == "food:new"
    assert "semantic" not in results[0].matched_axes

    mem.close()


def test_context_uses_semantic_supporting_events_in_known_mode(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"food\": \"noodle-soup\"}, \"id\": \"food:ramen\", \"type\": \"food\"} manual": [1.0, 0.0, 0.0],
            }
        ),
    )
    mem.append(
        "entity.create",
        {"id": "food:ramen", "type": "food", "attrs": {"food": "noodle-soup"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.flush("index")

    text = mem.context("ramen", max_tokens=300)

    assert "## Memory Basis" in text
    assert "food:ramen" in text
    assert "noodle-soup" in text

    mem.close()


def test_semantic_search_respects_known_time_visibility_after_reprocess(tmp_path, monkeypatch):
    first = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(
            version="extractor-v1",
            events=[
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "food:item", "type": "food", "attrs": {"name": "noodle-soup"}},
                    source_role="user",
                )
            ],
        ),
        embedder=StaticEmbedder(
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"name\": \"noodle-soup\"}, \"id\": \"food:item\", \"type\": \"food\"} user": [1.0, 0.0, 0.0],
            },
        ),
    )
    ack = first.turn(
        user="I like noodle soup",
        assistant="Noted.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    monkeypatch.setattr(canonical_module, "utcnow", lambda: dt("2026-05-01T10:00:01Z"))
    first.flush("canonical")
    first.flush("index")
    first.close()

    second = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(
            version="extractor-v2",
            events=[
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "food:item", "type": "food", "attrs": {"name": "broth-bowl"}},
                    source_role="user",
                )
            ],
        ),
        embedder=StaticEmbedder(
            version="embed-v2",
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"name\": \"noodle-soup\"}, \"id\": \"food:item\", \"type\": \"food\"} user": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"name\": \"broth-bowl\"}, \"id\": \"food:item\", \"type\": \"food\"} user": [0.0, 1.0, 0.0],
            },
        ),
    )
    monkeypatch.setattr(canonical_module, "utcnow", lambda: dt("2026-05-02T09:00:00Z"))
    second.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    second.flush("index")

    old_known = second.search("ramen", k=5, time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-01T10:00:01Z")))
    current_known = second.search("ramen", k=5)

    assert old_known
    assert old_known[0].entity_id == "food:item"
    assert "semantic" in old_known[0].matched_axes
    assert current_known == []

    second.close()


def test_semantic_search_valid_mode_uses_only_active_run_events(tmp_path):
    first = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(
            version="extractor-v1",
            events=[
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "food:item", "type": "food", "attrs": {"name": "noodle-soup"}},
                    effective_at_start=dt("2026-05-01T00:00:00Z"),
                    source_role="user",
                    time_confidence="exact",
                )
            ],
        ),
        embedder=StaticEmbedder(
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"name\": \"noodle-soup\"}, \"id\": \"food:item\", \"type\": \"food\"} user": [1.0, 0.0, 0.0],
            },
        ),
    )
    ack = first.turn(
        user="I like noodle soup",
        assistant="Noted.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    first.flush("canonical")
    first.flush("index")
    first.close()

    second = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=StaticExtractor(
            version="extractor-v2",
            events=[
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "food:item", "type": "food", "attrs": {"name": "broth-bowl"}},
                    effective_at_start=dt("2026-05-01T00:00:00Z"),
                    source_role="user",
                    time_confidence="exact",
                )
            ],
        ),
        embedder=StaticEmbedder(
            version="embed-v2",
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"name\": \"noodle-soup\"}, \"id\": \"food:item\", \"type\": \"food\"} user": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"name\": \"broth-bowl\"}, \"id\": \"food:item\", \"type\": \"food\"} user": [0.0, 1.0, 0.0],
            },
        ),
    )
    second.reprocess(from_turn_id=ack.turn_id, to_turn_id=ack.turn_id)
    second.flush("index")

    results = second.search(
        "ramen",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-02T00:00:00Z")),
        k=5,
    )

    assert results == []

    second.close()


def test_context_uses_semantic_supporting_events_in_valid_mode(tmp_path):
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        embedder=StaticEmbedder(
            mapping={
                "ramen": [1.0, 0.0, 0.0],
                "entity.create {\"attrs\": {\"food\": \"noodle-soup\"}, \"id\": \"food:ramen\", \"type\": \"food\"} manual": [1.0, 0.0, 0.0],
            }
        ),
    )
    mem.append(
        "entity.create",
        {"id": "food:ramen", "type": "food", "attrs": {"food": "noodle-soup"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.flush("index")

    text = mem.context(
        "ramen",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-02T00:00:00Z")),
        max_tokens=300,
    )

    assert "## Memory Basis" in text
    assert "- mode: valid" in text
    assert "food:ramen" in text
    assert "noodle-soup" in text

    mem.close()
