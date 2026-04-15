from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from .storage.store import EventStore, covers_valid_time, overlaps_valid_time_window, valid_event_sort_key
from .time_utils import to_rfc3339, utcnow
from .types import Event, SearchResult

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣:_-]+")
_KOREAN_SUFFIXES = (
    "으로는",
    "으로도",
    "으로",
    "에서",
    "에게",
    "한테",
    "이랑",
    "처럼",
    "까지",
    "부터",
    "보다",
    "에는",
    "에도",
    "으로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "로",
    "도",
    "만",
    "야",
)


@dataclass(slots=True)
class ScoredEvent:
    event: Event
    score: float


@dataclass(frozen=True, slots=True)
class QueryToken:
    raw: str
    variants: tuple[str, ...]


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
        events = self.store.visible_events_known(
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

    def search_valid(
        self,
        query: str,
        *,
        k: int,
        time_window: tuple[datetime, datetime] | None = None,
    ) -> list[SearchResult]:
        tokens = _query_tokens(query)
        if not tokens:
            return []

        if time_window is None:
            as_of = utcnow()
            events = [
                event
                for event in self.store.visible_events_valid()
                if covers_valid_time(event, as_of)
            ]
        else:
            start_at, end_at = time_window
            events = [
                event
                for event in self.store.visible_events_valid()
                if overlaps_valid_time_window(event, start_at, end_at)
            ]

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
                key=lambda item: (-item.score, *valid_event_sort_key(item.event)),
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
                    time_basis="valid",
                )
            )

        results.sort(key=lambda item: (-item.score, item.entity_id))
        return results[:k]


def _query_tokens(query: str) -> list[QueryToken]:
    tokens: list[QueryToken] = []
    for raw_token in _TOKEN_RE.findall(query):
        token = raw_token.lower().strip()
        if not token:
            continue
        variants = [token]
        if _contains_hangul(token):
            for suffix in _KOREAN_SUFFIXES:
                if token.endswith(suffix):
                    stem = token[: -len(suffix)]
                    if len(stem) >= 2:
                        variants.append(stem)
        deduped = tuple(dict.fromkeys(variants))
        tokens.append(QueryToken(raw=token, variants=deduped))
    return tokens


def _score_event(event: Event, tokens: list[QueryToken]) -> ScoredEvent:
    haystack = _event_haystack(event)
    matched = [
        token
        for token in tokens
        if any(variant in haystack for variant in token.variants)
    ]
    if not matched:
        return ScoredEvent(event=event, score=0.0)

    score = len(matched) / max(len(tokens), 1)
    phrase = " ".join(token.raw for token in tokens)
    if phrase and phrase in haystack:
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


def _contains_hangul(value: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in value)
