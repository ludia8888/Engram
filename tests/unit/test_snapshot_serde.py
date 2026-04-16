from __future__ import annotations

import pytest

from engram.snapshot_serde import deserialize_snapshot, serialize_snapshot
from engram.types import Entity, RelationEdge

from tests.conftest import dt


def _sample_entities() -> dict[str, Entity]:
    return {
        "user:alice": Entity(
            id="user:alice",
            type="user",
            attrs={"diet": "vegetarian", "location": "Busan"},
            created_recorded_at=dt("2026-05-01T10:00:00Z"),
            updated_recorded_at=dt("2026-05-01T11:00:00Z"),
        ),
    }


def _sample_relations() -> dict[str, tuple[RelationEdge, ...]]:
    return {
        "user:alice": (
            RelationEdge(
                relation_type="manager",
                other_entity_id="person:bob",
                direction="outgoing",
                attrs={"scope": "work"},
            ),
        ),
    }


def test_round_trip_entities_and_relations():
    entities = _sample_entities()
    relations = _sample_relations()
    state_blob, relation_blob = serialize_snapshot(entities, relations)

    restored_entities, restored_relations = deserialize_snapshot(state_blob, relation_blob)

    assert restored_entities["user:alice"].id == "user:alice"
    assert restored_entities["user:alice"].attrs == {"diet": "vegetarian", "location": "Busan"}
    assert restored_entities["user:alice"].created_recorded_at == dt("2026-05-01T10:00:00Z")
    assert len(restored_relations["user:alice"]) == 1
    assert restored_relations["user:alice"][0].relation_type == "manager"
    assert restored_relations["user:alice"][0].attrs == {"scope": "work"}


def test_round_trip_empty_snapshot():
    state_blob, relation_blob = serialize_snapshot({}, {})
    entities, relations = deserialize_snapshot(state_blob, relation_blob)
    assert entities == {}
    assert relations == {}


def test_corrupt_blob_raises_value_error():
    with pytest.raises(ValueError, match="corrupt"):
        deserialize_snapshot(b"not-gzip", b"not-gzip")
