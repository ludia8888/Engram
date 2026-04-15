from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from engram.types import Event, RelationEdge

from .temporal import covers_valid_time, valid_event_sort_key


def fold_relation_edges(
    entity_id: str,
    events: list[Event],
    *,
    endpoint_active=None,
) -> list[RelationEdge]:
    active_relations: dict[tuple[str, str, str], dict[str, Any]] = {}

    for event in events:
        if not event.type.startswith("relation."):
            continue

        source = str(event.data["source"])
        target = str(event.data["target"])
        relation_type = str(event.data["type"])
        if entity_id not in {source, target}:
            continue

        key = (source, target, relation_type)
        if event.type == "relation.create":
            active_relations[key] = dict(event.data.get("attrs", {}))
        elif event.type == "relation.update":
            current = dict(active_relations.get(key, {}))
            current.update(event.data.get("attrs", {}))
            active_relations[key] = current
        elif event.type == "relation.delete":
            active_relations.pop(key, None)

    edges: list[RelationEdge] = []
    for (source, target, relation_type), attrs in active_relations.items():
        if endpoint_active is not None and (not endpoint_active(source) or not endpoint_active(target)):
            continue
        if entity_id == source:
            edges.append(
                RelationEdge(
                    relation_type=relation_type,
                    other_entity_id=target,
                    direction="outgoing",
                    attrs=dict(attrs),
                )
            )
        if entity_id == target:
            edges.append(
                RelationEdge(
                    relation_type=relation_type,
                    other_entity_id=source,
                    direction="incoming",
                    attrs=dict(attrs),
                )
            )

    edges.sort(key=lambda edge: (edge.direction, edge.relation_type, edge.other_entity_id))
    return edges


def fold_relation_edges_in_window(
    entity_id: str,
    events: list[Event],
    start_at: datetime,
    end_at: datetime,
    *,
    endpoint_active_in_window=None,
) -> list[RelationEdge]:
    active_relations = relation_window_states(
        events,
        start_at,
        end_at,
        endpoint_active_in_window=endpoint_active_in_window,
    )

    edges: list[RelationEdge] = []
    for (source, target, relation_type), attrs in active_relations.items():
        if entity_id == source:
            edges.append(
                RelationEdge(
                    relation_type=relation_type,
                    other_entity_id=target,
                    direction="outgoing",
                    attrs=dict(attrs),
                )
            )
        if entity_id == target:
            edges.append(
                RelationEdge(
                    relation_type=relation_type,
                    other_entity_id=source,
                    direction="incoming",
                    attrs=dict(attrs),
                )
            )

    edges.sort(key=lambda edge: (edge.direction, edge.relation_type, edge.other_entity_id))
    return edges


def relation_window_states(
    events: list[Event],
    start_at: datetime,
    end_at: datetime,
    *,
    endpoint_active_in_window=None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    current_relations: dict[tuple[str, str, str], tuple[datetime, dict[str, Any]]] = {}
    overlapping_relations: dict[tuple[str, str, str], dict[str, Any]] = {}

    for event in sorted(events, key=valid_event_sort_key):
        if not event.type.startswith("relation.") or event.effective_at_start is None:
            continue

        source = str(event.data["source"])
        target = str(event.data["target"])
        relation_type = str(event.data["type"])
        key = (source, target, relation_type)
        event_start = event.effective_at_start

        if event.type in {"relation.create", "relation.update"}:
            if key in current_relations:
                current_start, current_attrs = current_relations[key]
                _capture_relation_window_overlap(
                    key,
                    current_start,
                    event_start,
                    current_attrs,
                    start_at,
                    end_at,
                    overlapping_relations,
                    endpoint_active_in_window=endpoint_active_in_window,
                )
                next_attrs = dict(current_attrs) if event.type == "relation.update" else {}
            else:
                next_attrs = {}
            next_attrs.update(event.data.get("attrs", {}))
            current_relations[key] = (event_start, next_attrs)
            continue

        if event.type == "relation.delete" and key in current_relations:
            current_start, current_attrs = current_relations.pop(key)
            _capture_relation_window_overlap(
                key,
                current_start,
                event_start,
                current_attrs,
                start_at,
                end_at,
                overlapping_relations,
                endpoint_active_in_window=endpoint_active_in_window,
            )

    for key, (current_start, current_attrs) in current_relations.items():
        _capture_relation_window_overlap(
            key,
            current_start,
            None,
            current_attrs,
            start_at,
            end_at,
            overlapping_relations,
            endpoint_active_in_window=endpoint_active_in_window,
        )

    return overlapping_relations


def _capture_relation_window_overlap(
    key: tuple[str, str, str],
    relation_start: datetime,
    relation_end: datetime | None,
    attrs: dict[str, Any],
    window_start: datetime,
    window_end: datetime,
    overlapping_relations: dict[tuple[str, str, str], dict[str, Any]],
    *,
    endpoint_active_in_window=None,
) -> None:
    overlap_start = max(relation_start, window_start)
    overlap_end = min_optional(window_end, relation_end)
    if overlap_start >= overlap_end:
        return
    if endpoint_active_in_window is not None:
        source, target, _relation_type = key
        if not endpoint_active_in_window(source, target, overlap_start, overlap_end):
            return
    overlapping_relations[key] = dict(attrs)


def entity_active_intervals(
    entity_id: str,
    events: list[Event],
) -> list[tuple[datetime, datetime | None]]:
    intervals: list[tuple[datetime, datetime | None]] = []
    active = False
    current_start: datetime | None = None

    for event in sorted(events, key=valid_event_sort_key):
        if not event.type.startswith("entity.") or event.data["id"] != entity_id:
            continue
        if event.effective_at_start is None:
            continue
        if event.type == "entity.create":
            if active and current_start is not None:
                intervals.append((current_start, event.effective_at_start))
            active = True
            current_start = event.effective_at_start
        elif event.type == "entity.update":
            if not active:
                active = True
                current_start = event.effective_at_start
        elif event.type == "entity.delete":
            if active and current_start is not None:
                intervals.append((current_start, event.effective_at_start))
            active = False
            current_start = None

    if active and current_start is not None:
        intervals.append((current_start, None))
    return intervals


def intervals_overlap(
    start_a: datetime,
    end_a: datetime | None,
    start_b: datetime,
    end_b: datetime | None,
) -> bool:
    overlap_start = max(start_a, start_b)
    overlap_end = min_optional(end_a, end_b)
    return overlap_start < overlap_end


def min_optional(*values: datetime | None) -> datetime:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return datetime.max.replace(tzinfo=UTC)
    return min(filtered)
