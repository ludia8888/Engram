from __future__ import annotations

import math
from datetime import datetime

from .storage.store import EventStore
from .time_utils import to_rfc3339, utcnow
from .types import Event, RawTurn, SearchResult, TemporalEntityView


class ContextBuilder:
    def __init__(self, store: EventStore, raw_get):
        self.store = store
        self.raw_get = raw_get

    def build_known(
        self,
        *,
        query: str,
        results: list[SearchResult],
        as_of: datetime | None,
        max_tokens: int,
        include_history: bool,
        include_raw: bool,
        get_known_at,
    ) -> str:
        return self._build(
            mode="known",
            query=query,
            results=results,
            as_of=as_of,
            max_tokens=max_tokens,
            include_history=include_history,
            include_raw=include_raw,
            get_view=get_known_at,
        )

    def build_valid(
        self,
        *,
        query: str,
        results: list[SearchResult],
        as_of: datetime | None,
        max_tokens: int,
        include_history: bool,
        include_raw: bool,
        get_valid_at,
    ) -> str:
        return self._build(
            mode="valid",
            query=query,
            results=results,
            as_of=as_of,
            max_tokens=max_tokens,
            include_history=include_history,
            include_raw=include_raw,
            get_view=get_valid_at,
        )

    def _build(
        self,
        *,
        mode: str,
        query: str,
        results: list[SearchResult],
        as_of: datetime | None,
        max_tokens: int,
        include_history: bool,
        include_raw: bool,
        get_view,
    ) -> str:
        basis_time = as_of or utcnow()
        supporting_events = self.store.events_by_ids(_supporting_event_ids(results))
        sections: list[list[str]] = [
            [
                "## Memory Basis",
                f"- mode: {mode}",
                f"- as_of: {to_rfc3339(basis_time)}",
                f"- query: {query}",
            ]
        ]

        current_state = ["## Current State"]
        for result in results:
            view: TemporalEntityView | None = get_view(result.entity_id, basis_time)
            if view is None:
                continue
            line = f"- {view.entity_id} ({view.entity_type}) attrs={view.attrs}"
            if view.unknown_attrs:
                line += f" unknown_attrs={view.unknown_attrs}"
            current_state.append(line)
        if len(current_state) > 1:
            sections.append(current_state)

        if include_history:
            change_lines = ["## Relevant Changes"]
            for event in supporting_events:
                time_label = _event_time_label(event, mode)
                if event.type == "entity.delete":
                    change_lines.append(
                        f"- {event.data['id']} deleted {time_label}"
                    )
                    continue
                attrs = event.data.get("attrs", {})
                if not attrs:
                    continue
                change_lines.append(
                    f"- {event.data['id']} {attrs} {time_label} "
                    f"confidence={event.confidence if event.confidence is not None else 'unknown'} "
                    f"reason={event.reason or 'n/a'}"
                )
            if len(change_lines) > 1:
                sections.append(change_lines)

        if include_raw:
            raw_lines = ["## Raw Evidence"]
            for turn in self._raw_turns_for_events(supporting_events):
                raw_lines.append(f'- [{to_rfc3339(turn.observed_at)}] "{turn.user}"')
            if len(raw_lines) > 1:
                sections.append(raw_lines)

        return _truncate_sections(sections, max_tokens)

    def _raw_turns_for_events(self, events: list[Event]) -> list[RawTurn]:
        turns: list[RawTurn] = []
        seen_turn_ids: set[str] = set()
        for event in events:
            if event.source_turn_id is None or event.source_turn_id in seen_turn_ids:
                continue
            turn = self.raw_get(event.source_turn_id)
            if turn is None:
                continue
            seen_turn_ids.add(turn.id)
            turns.append(turn)
        return turns


def _supporting_event_ids(results: list[SearchResult]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for result in results:
        for event_id in result.supporting_event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            ordered.append(event_id)
    return ordered


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _event_time_label(event: Event, mode: str) -> str:
    if mode == "valid":
        if event.effective_at_start is None:
            return "effective_at=unknown"
        if event.effective_at_end is None:
            return f"effective_at={to_rfc3339(event.effective_at_start)}"
        return (
            f"effective_at={to_rfc3339(event.effective_at_start)}"
            f"..{to_rfc3339(event.effective_at_end)}"
        )
    return f"recorded_at={to_rfc3339(event.recorded_at)}"


def _truncate_sections(sections: list[list[str]], max_tokens: int) -> str:
    lines: list[str] = []
    used = 0
    for section in sections:
        if not section:
            continue
        if lines:
            candidate = "\n"
            if used + _estimate_tokens(candidate) > max_tokens:
                break
            lines.append("")
            used += _estimate_tokens(candidate)
        for line in section:
            tokens = _estimate_tokens(line)
            if used + tokens > max_tokens:
                return "\n".join(lines).strip()
            lines.append(line)
            used += tokens
    return "\n".join(lines).strip()
