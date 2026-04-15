from __future__ import annotations

from datetime import UTC, datetime

from .errors import ValidationError


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime | None, field_name: str) -> datetime:
    if value is None:
        return utcnow()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field_name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def to_rfc3339(value: datetime) -> str:
    return ensure_utc(value, "datetime").isoformat().replace("+00:00", "Z")


def from_rfc3339(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    return ensure_utc(parsed, "datetime")

