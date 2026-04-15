from __future__ import annotations

import math
from datetime import datetime

from .storage.store import EventStore
from .time_utils import to_rfc3339, utcnow
from .types import RawTurn, SearchResult, TemporalEntityView


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
        basis_time = as_of or utcnow()
        sections: list[list[str]] = [
            [
                "## Memory Basis",
                "- mode: known",
                f"- as_of: {to_rfc3339(basis_time)}",
                f"- query: {query}",
            ]
        ]

        current_state = ["## Current State"]
        for result in results:
            view: TemporalEntityView | None = get_known_at(result.entity_id, basis_time)
            if view is None:
                continue
            current_state.append(f"- {view.entity_id} ({view.entity_type}) attrs={view.attrs}")
        if len(current_state) > 1:
            sections.append(current_state)

        if include_history:
            change_lines = ["## Relevant Changes"]
            for event in self.store.events_by_ids(_supporting_event_ids(results)):
                if event.type == "entity.delete":
                    change_lines.append(
                        f"- {event.data['id']} deleted at {to_rfc3339(event.recorded_at)}"
                    )
                    continue
                attrs = event.data.get("attrs", {})
                if not attrs:
                    continue
                change_lines.append(
                    f"- {event.data['id']} {attrs} recorded_at={to_rfc3339(event.recorded_at)} "
                    f"confidence={event.confidence if event.confidence is not None else 'unknown'} "
                    f"reason={event.reason or 'n/a'}"
                )
            if len(change_lines) > 1:
                sections.append(change_lines)

        if include_raw:
            raw_lines = ["## Raw Evidence"]
            for turn in self._raw_turns_for_results(results):
                raw_lines.append(f'- [{to_rfc3339(turn.observed_at)}] "{turn.user}"')
            if len(raw_lines) > 1:
                sections.append(raw_lines)

        return _truncate_sections(sections, max_tokens)

    def _raw_turns_for_results(self, results: list[SearchResult]) -> list[RawTurn]:
        turns: list[RawTurn] = []
        seen_turn_ids: set[str] = set()
        for event in self.store.events_by_ids(_supporting_event_ids(results)):
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
