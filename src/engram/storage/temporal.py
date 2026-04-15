from __future__ import annotations

from datetime import UTC, datetime

from engram.types import Event


def covers_valid_time(event: Event, at: datetime) -> bool:
    if _has_unknown_effective_time(event):
        return False
    start = event.effective_at_start
    end = event.effective_at_end
    if start is None:
        return False
    if start > at:
        return False
    if end is not None and at >= end:
        return False
    return True


def overlaps_valid_time_window(
    event: Event,
    start_at: datetime,
    end_at: datetime,
) -> bool:
    if _has_unknown_effective_time(event):
        return False

    start = event.effective_at_start
    end = event.effective_at_end
    if start is None:
        return False
    if end is not None and end <= start_at:
        return False
    if start >= end_at:
        return False
    return True


def valid_event_sort_key(event: Event) -> tuple[bool, datetime, datetime, int]:
    return (
        event.effective_at_start is None,
        event.effective_at_start or datetime.max.replace(tzinfo=UTC),
        event.recorded_at,
        event.seq,
    )


def _has_unknown_effective_time(event: Event) -> bool:
    return event.effective_at_start is None
