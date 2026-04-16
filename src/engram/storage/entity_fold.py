from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engram.types import Event

from .temporal import _has_unknown_effective_time, covers_valid_time, valid_event_sort_key


@dataclass(slots=True)
class FoldedEntityState:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    supporting_event_ids: list[str]
    created_recorded_at: datetime | None
    updated_recorded_at: datetime | None
    active: bool


@dataclass(slots=True)
class FoldedValidEntityState:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    unknown_attrs: list[str]
    supporting_event_ids: list[str]
    active: bool


def fold_entity_events(entity_id: str, events: list[Event]) -> FoldedEntityState | None:
    entity_type = "unknown"
    attrs: dict[str, Any] = {}
    supporting_event_ids: list[str] = []
    created_at = None
    updated_at = None
    active = False

    for event in events:
        if not event.type.startswith("entity.") or event.data["id"] != entity_id:
            continue
        supporting_event_ids.append(event.id)
        if event.type == "entity.create":
            entity_type = event.data["type"]
            attrs = dict(event.data["attrs"])
            created_at = event.recorded_at
            updated_at = event.recorded_at
            active = True
        elif event.type == "entity.update":
            if not active:
                created_at = event.recorded_at
                active = True
            attrs.update(event.data["attrs"])
            updated_at = event.recorded_at
        elif event.type == "entity.delete":
            attrs = {}
            created_at = None
            updated_at = None
            active = False

    if not active or created_at is None or updated_at is None:
        return None

    return FoldedEntityState(
        entity_id=entity_id,
        entity_type=entity_type,
        attrs=dict(attrs),
        supporting_event_ids=supporting_event_ids,
        created_recorded_at=created_at,
        updated_recorded_at=updated_at,
        active=active,
    )


def fold_entity_events_valid_at(
    entity_id: str,
    at: datetime,
    events: list[Event],
) -> FoldedValidEntityState | None:
    events = sorted(events, key=valid_event_sort_key)
    entity_type = "unknown"
    attrs: dict[str, Any] = {}
    unknown_attrs: list[str] = []
    supporting_event_ids: list[str] = []
    active = False

    for event in events:
        if not event.type.startswith("entity.") or event.data["id"] != entity_id:
            continue
        if not covers_valid_time(event, at):
            if _has_unknown_effective_time(event):
                unknown_attrs = _merge_unknown_attrs(unknown_attrs, event.data.get("attrs", {}).keys())
            continue

        supporting_event_ids.append(event.id)
        if event.type == "entity.create":
            entity_type = event.data["type"]
            attrs = dict(event.data["attrs"])
            active = True
        elif event.type == "entity.update":
            if not active:
                active = True
            attrs.update(event.data["attrs"])
        elif event.type == "entity.delete":
            attrs = {}
            active = False
            unknown_attrs = []

    if not active and not unknown_attrs:
        return None

    return FoldedValidEntityState(
        entity_id=entity_id,
        entity_type=entity_type,
        attrs=dict(attrs),
        unknown_attrs=unknown_attrs,
        supporting_event_ids=supporting_event_ids,
        active=active,
    )


def _merge_unknown_attrs(existing: list[str], new_keys) -> list[str]:
    seen = set(existing)
    merged = list(existing)
    for key in new_keys:
        if key in seen:
            continue
        seen.add(key)
        merged.append(str(key))
    return merged
