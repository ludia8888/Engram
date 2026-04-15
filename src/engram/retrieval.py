from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

from .semantic import Embedder, cosine_similarity
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
    lexical_score: float
    semantic_score: float
    combined_score: float


@dataclass(frozen=True, slots=True)
class QueryToken:
    raw: str
    variants: tuple[str, ...]


VisibleEventsProvider = Callable[[tuple[datetime, datetime] | None], list[Event]]
EventSortKey = Callable[[ScoredEvent], tuple]


_SEMANTIC_WEIGHT = 0.4
_LEXICAL_WEIGHT = 0.6
_SEMANTIC_MIN_SCORE = 0.35


class RetrievalEngine:
    def __init__(self, store: EventStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def search_known(
        self,
        query: str,
        *,
        k: int,
        time_window: tuple[datetime, datetime] | None = None,
    ) -> list[SearchResult]:
        return self._search(
            query,
            k=k,
            time_mode="known",
            time_window=time_window,
            visible_events_provider=self._known_visible_events,
            sort_key=lambda item: (-item.combined_score, item.event.recorded_at, item.event.seq),
        )

    def search_valid(
        self,
        query: str,
        *,
        k: int,
        time_window: tuple[datetime, datetime] | None = None,
    ) -> list[SearchResult]:
        return self._search(
            query,
            k=k,
            time_mode="valid",
            time_window=time_window,
            visible_events_provider=self._valid_visible_events,
            sort_key=lambda item: (-item.combined_score, *valid_event_sort_key(item.event)),
        )

    def _search(
        self,
        query: str,
        *,
        k: int,
        time_mode: Literal["known", "valid"],
        time_window: tuple[datetime, datetime] | None,
        visible_events_provider: VisibleEventsProvider,
        sort_key: EventSortKey,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        tokens = _query_tokens(query)
        events = visible_events_provider(time_window)
        if not events:
            return []

        lexical_scores = {
            event.id: _score_event_lexical(event, tokens)
            for event in events
        }
        semantic_scores = self._semantic_scores(query, events)

        scored_events: list[ScoredEvent] = []
        for event in events:
            lexical_score = lexical_scores.get(event.id, 0.0)
            semantic_score = semantic_scores.get(event.id, 0.0)
            if semantic_score > 0:
                combined = (
                    (_LEXICAL_WEIGHT * lexical_score if lexical_score > 0 else 0.0)
                    + (_SEMANTIC_WEIGHT * semantic_score if semantic_score > 0 else 0.0)
                )
            else:
                combined = lexical_score
            if lexical_score <= 0 and semantic_score <= 0:
                continue
            if combined <= 0:
                continue
            scored_events.append(
                ScoredEvent(
                    event=event,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    combined_score=combined,
                )
            )

        if not scored_events:
            return []

        event_entities = self.store.event_entity_ids_for_events([scored.event.id for scored in scored_events])
        entity_scores: dict[str, float] = defaultdict(float)
        entity_event_scores: dict[str, list[ScoredEvent]] = defaultdict(list)

        for scored in scored_events:
            for entity_id in event_entities.get(scored.event.id, []):
                entity_scores[entity_id] += scored.combined_score
                entity_event_scores[entity_id].append(scored)

        results: list[SearchResult] = []
        for entity_id, total_score in entity_scores.items():
            ranked_events = sorted(entity_event_scores[entity_id], key=sort_key)
            matched_axes: set[str] = {"entity"}
            if any(item.semantic_score > 0 for item in ranked_events):
                matched_axes.add("semantic")
            if time_window is not None:
                matched_axes.add("temporal")
            results.append(
                SearchResult(
                    entity_id=entity_id,
                    score=round(total_score, 6),
                    matched_axes=matched_axes,
                    supporting_event_ids=[item.event.id for item in ranked_events[:5]],
                    time_basis=time_mode,
                )
            )

        results.sort(key=lambda item: (-item.score, item.entity_id))
        return results[:k]

    def _known_visible_events(self, time_window: tuple[datetime, datetime] | None) -> list[Event]:
        upper_bound = time_window[1] if time_window else utcnow()
        lower_bound = time_window[0] if time_window else None
        events = self.store.visible_events_known(
            to_rfc3339(upper_bound),
            from_recorded_at=to_rfc3339(lower_bound) if lower_bound else None,
        )
        recorded_at = to_rfc3339(upper_bound)
        return [
            event
            for event in events
            if self.store.relation_event_is_live_known(event, recorded_at)
        ]

    def _valid_visible_events(self, time_window: tuple[datetime, datetime] | None) -> list[Event]:
        visible_events = self.store.visible_events_valid()
        if time_window is None:
            as_of = utcnow()
            return [
                event
                for event in visible_events
                if covers_valid_time(event, as_of) and self.store.relation_event_is_live_valid(event, as_of)
            ]

        start_at, end_at = time_window
        return [
            event
            for event in visible_events
            if overlaps_valid_time_window(event, start_at, end_at)
            and self.store.relation_event_is_live_valid(event, end_at)
        ]

    def _semantic_scores(self, query: str, events: list[Event]) -> dict[str, float]:
        event_embeddings = self.store.event_embeddings_for_ids(
            [event.id for event in events],
            embedder_version=self.embedder.version,
        )
        if not event_embeddings:
            return {}

        query_embedding = self.embedder.embed_texts([query])
        if len(query_embedding) != 1:
            raise ValueError(f"embedder returned {len(query_embedding)} query embeddings")

        scores: dict[str, float] = {}
        query_vector = query_embedding[0]
        for event in events:
            event_vector = event_embeddings.get(event.id)
            if event_vector is None:
                continue
            score = max(0.0, cosine_similarity(query_vector, event_vector))
            if score >= _SEMANTIC_MIN_SCORE:
                scores[event.id] = score
        return scores


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


def _score_event_lexical(event: Event, tokens: list[QueryToken]) -> float:
    if not tokens:
        return 0.0
    haystack = _event_haystack(event)
    matched = [
        token
        for token in tokens
        if any(variant in haystack for variant in token.variants)
    ]
    if not matched:
        return 0.0

    score = len(matched) / max(len(tokens), 1)
    phrase = " ".join(token.raw for token in tokens)
    if phrase and phrase in haystack:
        score += 0.25
    return score


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
