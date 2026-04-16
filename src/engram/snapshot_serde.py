from __future__ import annotations

import gzip
import json
from typing import Any, Mapping

from .time_utils import from_rfc3339, to_rfc3339
from .types import Entity, RelationEdge


def serialize_snapshot(
    entities: Mapping[str, Entity],
    relations: Mapping[str, tuple[RelationEdge, ...]],
) -> tuple[bytes, bytes]:
    state = {
        entity_id: _entity_to_dict(entity)
        for entity_id, entity in entities.items()
    }
    rels = {
        entity_id: [_relation_edge_to_dict(edge) for edge in edges]
        for entity_id, edges in relations.items()
    }
    state_blob = gzip.compress(json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    relation_blob = gzip.compress(json.dumps(rels, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return state_blob, relation_blob


def deserialize_snapshot(
    state_blob: bytes,
    relation_blob: bytes,
) -> tuple[dict[str, Entity], dict[str, tuple[RelationEdge, ...]]]:
    try:
        state_raw = json.loads(gzip.decompress(state_blob).decode("utf-8"))
        rels_raw = json.loads(gzip.decompress(relation_blob).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"corrupt snapshot blob: {exc}") from exc

    entities: dict[str, Entity] = {}
    for entity_id, data in state_raw.items():
        entities[entity_id] = _dict_to_entity(data)

    relations: dict[str, tuple[RelationEdge, ...]] = {}
    for entity_id, edges in rels_raw.items():
        relations[entity_id] = tuple(_dict_to_relation_edge(edge) for edge in edges)

    return entities, relations


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "type": entity.type,
        "attrs": entity.attrs,
        "created_recorded_at": to_rfc3339(entity.created_recorded_at),
        "updated_recorded_at": to_rfc3339(entity.updated_recorded_at),
    }


def _dict_to_entity(data: dict[str, Any]) -> Entity:
    return Entity(
        id=data["id"],
        type=data["type"],
        attrs=data["attrs"],
        created_recorded_at=from_rfc3339(data["created_recorded_at"]),
        updated_recorded_at=from_rfc3339(data["updated_recorded_at"]),
    )


def _relation_edge_to_dict(edge: RelationEdge) -> dict[str, Any]:
    return {
        "relation_type": edge.relation_type,
        "other_entity_id": edge.other_entity_id,
        "direction": edge.direction,
        "attrs": edge.attrs,
    }


def _dict_to_relation_edge(data: dict[str, Any]) -> RelationEdge:
    return RelationEdge(
        relation_type=data["relation_type"],
        other_entity_id=data["other_entity_id"],
        direction=data["direction"],
        attrs=data["attrs"],
    )
