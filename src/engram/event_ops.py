from __future__ import annotations

from uuid import uuid4

from .errors import ValidationError
from .time_utils import to_rfc3339, utcnow
from .types import Event


def validate_event(event_type: str, data: dict) -> None:
    if event_type == "entity.create":
        if not isinstance(data.get("id"), str) or not isinstance(data.get("type"), str):
            raise ValidationError("entity.create requires string id and type")
        if not isinstance(data.get("attrs"), dict):
            raise ValidationError("entity.create requires attrs dict")
        return
    if event_type == "entity.update":
        if not isinstance(data.get("id"), str):
            raise ValidationError("entity.update requires string id")
        if not isinstance(data.get("attrs"), dict):
            raise ValidationError("entity.update requires attrs dict")
        return
    if event_type == "entity.delete":
        if not isinstance(data.get("id"), str):
            raise ValidationError("entity.delete requires string id")
        return
    if event_type in {"relation.create", "relation.update"}:
        raise ValidationError(f"{event_type} is planned but not implemented in Phase 1")
    if event_type == "relation.delete":
        raise ValidationError("relation.delete is planned but not implemented in Phase 1")
    raise ValidationError(f"unsupported event type: {event_type}")


def derive_event_entities(event: Event) -> list[tuple[str, str]]:
    if event.type.startswith("entity."):
        return [(event.data["id"], "subject")]
    return [
        (event.data["source"], "source"),
        (event.data["target"], "target"),
    ]


def derive_dirty_rows(
    event: Event,
    event_entities: list[tuple[str, str]],
) -> list[tuple[str, str, str, str | None, str, str]]:
    created_at = to_rfc3339(utcnow())
    from_recorded_at = to_rfc3339(event.recorded_at)
    from_effective_at = to_rfc3339(event.effective_at_start) if event.effective_at_start else None
    rows: list[tuple[str, str, str, str | None, str, str]] = []
    for entity_id, _role in event_entities:
        rows.append(
            (
                str(uuid4()),
                entity_id,
                from_recorded_at,
                from_effective_at,
                f"{event.type}:{event.id}",
                created_at,
            )
        )
    return rows
