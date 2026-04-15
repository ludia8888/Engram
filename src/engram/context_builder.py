from __future__ import annotations

import math
from datetime import datetime

from .storage.store import EventStore
from .time_utils import to_rfc3339, utcnow
from .types import Event, RawTurn, RelationEdge, SearchResult, TemporalEntityView


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
        time_window: tuple[datetime, datetime] | None = None,
        max_tokens: int,
        include_history: bool,
        include_raw: bool,
        get_known_at,
        get_known_relations_at,
    ) -> str:
        return self._build(
            mode="known",
            query=query,
            results=results,
            as_of=as_of,
            time_window=time_window,
            max_tokens=max_tokens,
            include_history=include_history,
            include_raw=include_raw,
            get_view=get_known_at,
            get_relations=get_known_relations_at,
        )

    def build_valid(
        self,
        *,
        query: str,
        results: list[SearchResult],
        as_of: datetime | None,
        time_window: tuple[datetime, datetime] | None,
        max_tokens: int,
        include_history: bool,
        include_raw: bool,
        get_valid_at,
        get_valid_relations_at,
        get_valid_relations_in_window,
    ) -> str:
        return self._build(
            mode="valid",
            query=query,
            results=results,
            as_of=as_of,
            time_window=time_window,
            max_tokens=max_tokens,
            include_history=include_history,
            include_raw=include_raw,
            get_view=get_valid_at,
            get_relations=get_valid_relations_at,
            get_relations_in_window=get_valid_relations_in_window,
        )

    def _build(
        self,
        *,
        mode: str,
        query: str,
        results: list[SearchResult],
        as_of: datetime | None,
        time_window: tuple[datetime, datetime] | None,
        max_tokens: int,
        include_history: bool,
        include_raw: bool,
        get_view,
        get_relations,
        get_relations_in_window=None,
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
        if any("causal" in result.matched_axes for result in results):
            sections[0].append("- causal_support: yes")
        if time_window is not None:
            sections[0].append(
                f"- time_window: {to_rfc3339(time_window[0])}..{to_rfc3339(time_window[1])}"
            )

        current_state = ["## Current State"]
        for result in results:
            view: TemporalEntityView | None = get_view(result.entity_id, basis_time)
            relation_summary_label = "relations"
            attrs_label = "attrs"
            unknown_attrs_label = "unknown_attrs"
            if time_window is not None and get_relations_in_window is not None:
                relations: list[RelationEdge] = list(
                    get_relations_in_window(
                        result.entity_id,
                        time_window[0],
                        time_window[1],
                    )
                )
                relation_summary_label = "relations_active_in_window"
                attrs_label = "attrs_as_of_window_end"
                unknown_attrs_label = "unknown_attrs_as_of_window_end"
            else:
                relations = list(get_relations(result.entity_id, basis_time))
            if view is None and not relations:
                continue
            entity_id = result.entity_id if view is None else view.entity_id
            entity_type = "unknown" if view is None else view.entity_type
            attrs = {} if view is None else view.attrs
            line = f"- {entity_id} ({entity_type}) {attrs_label}={attrs}"
            if view is not None and view.unknown_attrs:
                line += f" {unknown_attrs_label}={view.unknown_attrs}"
            if relations:
                line += f" {relation_summary_label}={_relation_summaries(relations)}"
            current_state.append(line)
        if len(current_state) > 1:
            sections.append(current_state)

        if include_history:
            change_lines = ["## Relevant Changes"]
            supporting_by_id = {event.id: event for event in supporting_events}
            downstream_by_id: dict[str, list[Event]] = {}
            for event in supporting_events:
                if event.caused_by is None:
                    continue
                downstream_by_id.setdefault(event.caused_by, []).append(event)
            for event in supporting_events:
                time_label = _event_time_label(event, mode)
                if event.type.startswith("relation."):
                    change_lines.append(
                        _describe_relation_event(
                            event,
                            time_label,
                            cause_event=supporting_by_id.get(event.caused_by) if event.caused_by else None,
                            downstream_events=downstream_by_id.get(event.id, []),
                        )
                    )
                    continue
                if event.type == "entity.delete":
                    line = f"- {event.data['id']} deleted {time_label}"
                    change_lines.append(
                        _append_causal_note(
                            line,
                            cause_event=supporting_by_id.get(event.caused_by) if event.caused_by else None,
                            downstream_events=downstream_by_id.get(event.id, []),
                        )
                    )
                    continue
                attrs = event.data.get("attrs", {})
                if not attrs:
                    continue
                line = (
                    f"- {event.data['id']} {attrs} {time_label} "
                    f"confidence={event.confidence if event.confidence is not None else 'unknown'} "
                    f"reason={event.reason or 'n/a'}"
                )
                change_lines.append(
                    _append_causal_note(
                        line,
                        cause_event=supporting_by_id.get(event.caused_by) if event.caused_by else None,
                        downstream_events=downstream_by_id.get(event.id, []),
                    )
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


def _relation_summaries(relations: list[RelationEdge]) -> list[str]:
    summaries: list[str] = []
    for relation in relations:
        if relation.direction == "outgoing":
            summary = f"{relation.relation_type} -> {relation.other_entity_id}"
        else:
            summary = f"{relation.relation_type} <- {relation.other_entity_id}"
        if relation.attrs:
            summary += f" attrs={relation.attrs}"
        summaries.append(summary)
    return summaries


def _describe_relation_event(
    event: Event,
    time_label: str,
    *,
    cause_event: Event | None = None,
    downstream_events: list[Event] | None = None,
) -> str:
    source = event.data["source"]
    target = event.data["target"]
    relation_type = event.data["type"]
    if event.type == "relation.delete":
        line = f"- relation {source} -[{relation_type}]-> {target} deleted {time_label}"
        return _append_causal_note(
            line,
            cause_event=cause_event,
            downstream_events=downstream_events or [],
        )
    attrs = event.data.get("attrs", {})
    line = (
        f"- relation {source} -[{relation_type}]-> {target} attrs={attrs} {time_label} "
        f"confidence={event.confidence if event.confidence is not None else 'unknown'} "
        f"reason={event.reason or 'n/a'}"
    )
    return _append_causal_note(
        line,
        cause_event=cause_event,
        downstream_events=downstream_events or [],
    )


def _append_causal_note(
    line: str,
    *,
    cause_event: Event | None,
    downstream_events: list[Event],
) -> str:
    notes: list[str] = []
    if cause_event is not None:
        notes.append(f"caused by: {_event_brief(cause_event)}")
    if downstream_events:
        led_to = ", ".join(_event_brief(event) for event in downstream_events[:2])
        notes.append(f"led to: {led_to}")
    if not notes:
        return line
    return f"{line} {'; '.join(notes)}"


def _event_brief(event: Event) -> str:
    if event.type.startswith("relation."):
        source = event.data["source"]
        target = event.data["target"]
        relation_type = event.data["type"]
        if event.type == "relation.delete":
            return f"relation {source} -[{relation_type}]-> {target} deleted"
        attrs = event.data.get("attrs", {})
        return f"relation {source} -[{relation_type}]-> {target} attrs={attrs}"
    if event.type == "entity.delete":
        return f"{event.data['id']} deleted"
    return f"{event.data['id']} {event.data.get('attrs', {})}"


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
