from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .storage.store import EventStore
from .time_utils import to_rfc3339, utcnow
from .types import Event, SearchResult

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣:_-]+")


@dataclass(slots=True)
class ScoredEvent:
    event: Event
    score: float


class RetrievalEngine:
    def __init__(self, store: EventStore):
        self.store = store

    def search_known(
        self,
        query: str,
        *,
        k: int,
        time_window: tuple[datetime, datetime] | None = None,
    ) -> list[SearchResult]:
        tokens = _query_tokens(query)
        if not tokens:
            return []

        upper_bound = time_window[1] if time_window else utcnow()
        lower_bound = time_window[0] if time_window else None
        events = self.store.visible_events(
            to_rfc3339(upper_bound),
            from_recorded_at=to_rfc3339(lower_bound) if lower_bound else None,
        )
        if not events:
            return []

        scored_events = [
            scored
            for scored in (_score_event(event, tokens) for event in events)
            if scored.score > 0
        ]
        if not scored_events:
            return []

        event_entities = self.store.event_entity_ids_for_events([scored.event.id for scored in scored_events])
        entity_scores: dict[str, float] = defaultdict(float)
        entity_event_scores: dict[str, list[ScoredEvent]] = defaultdict(list)

        for scored in scored_events:
            for entity_id in event_entities.get(scored.event.id, []):
                entity_scores[entity_id] += scored.score
                entity_event_scores[entity_id].append(scored)

        results: list[SearchResult] = []
        for entity_id, total_score in entity_scores.items():
            ranked_events = sorted(
                entity_event_scores[entity_id],
                key=lambda item: (-item.score, item.event.recorded_at, item.event.seq),
            )
            matched_axes: set[str] = {"entity"}
            if time_window is not None:
                matched_axes.add("temporal")
            results.append(
                SearchResult(
                    entity_id=entity_id,
                    score=round(total_score, 6),
                    matched_axes=matched_axes,
                    supporting_event_ids=[item.event.id for item in ranked_events[:5]],
                    time_basis="known",
                )
            )

        results.sort(key=lambda item: (-item.score, item.entity_id))
        return results[:k]


def _query_tokens(query: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(query) if token.strip()]


def _score_event(event: Event, tokens: Iterable[str]) -> ScoredEvent:
    haystack = _event_haystack(event)
    token_list = list(tokens)
    matched = [token for token in token_list if token in haystack]
    if not matched:
        return ScoredEvent(event=event, score=0.0)

    score = len(set(matched)) / max(len(set(token_list)), 1)
    if haystack.find(" ".join(token_list)) >= 0:
        score += 0.25
    return ScoredEvent(event=event, score=score)


def _event_haystack(event: Event) -> str:
    parts = [
        event.type,
        json.dumps(event.data, ensure_ascii=False, sort_keys=True),
        event.reason or "",
        event.source_role,
    ]
    return " ".join(parts).lower()
