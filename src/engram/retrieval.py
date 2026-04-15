from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
from typing import Callable, Literal

from .search_terms import QueryToken, event_search_text, query_candidate_terms, query_tokens
from .semantic import Embedder, cosine_similarity
from .storage.store import (
    EventStore,
    RelationWindowQueryCache,
    covers_valid_time,
    overlaps_valid_time_window,
    valid_event_sort_key,
)
from .time_utils import to_rfc3339, utcnow
from .types import Event, SearchResult


@dataclass(slots=True)
class ScoredEvent:
    event: Event
    lexical_score: float
    semantic_score: float
    direct_score: float
    causal_score: float = 0.0

    @property
    def combined_score(self) -> float:
        return self.direct_score + self.causal_score

VisibleEventsProvider = Callable[[tuple[datetime, datetime] | None, list[str] | None], list[Event]]
EventSortKey = Callable[[ScoredEvent], tuple]


_SEMANTIC_WEIGHT = 0.4
_LEXICAL_WEIGHT = 0.6
_SEMANTIC_MIN_SCORE = 0.35
_CAUSAL_WEIGHT = 0.5
_QUERY_EMBED_CACHE_SIZE = 128


class RetrievalEngine:
    def __init__(self, store: EventStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._query_embedding_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

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
        relation_window_cache: RelationWindowQueryCache | None = None,
    ) -> list[SearchResult]:
        return self._search(
            query,
            k=k,
            time_mode="valid",
            time_window=time_window,
            visible_events_provider=lambda window, candidate_ids: self._valid_visible_events(
                window,
                candidate_ids,
                relation_window_cache=relation_window_cache,
            ),
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

        tokens = query_tokens(query)
        lexical_candidate_ids = set(
            self.store.candidate_event_ids_for_search_terms(query_candidate_terms(query))
        )
        direct_events: list[Event]
        lexical_scores: dict[str, float]
        if lexical_candidate_ids:
            direct_events = visible_events_provider(time_window, sorted(lexical_candidate_ids))
            if not direct_events:
                return []
            lexical_scores = {
                event.id: (
                    _score_event_lexical(event, tokens)
                    if event.id in lexical_candidate_ids
                    else 0.0
                )
                for event in direct_events
            }
            lexical_hit_event_ids = [
                event.id
                for event in direct_events
                if lexical_scores.get(event.id, 0.0) > 0
            ]
            if lexical_hit_event_ids:
                lexical_hit_entity_ids = sorted(
                    {
                        entity_id
                        for entity_ids in self.store.event_entity_ids_for_events(
                            lexical_hit_event_ids
                        ).values()
                        for entity_id in entity_ids
                    }
                )
                semantic_candidate_ids = set(
                    self.store.event_ids_with_embeddings_for_entities(
                        lexical_hit_entity_ids,
                        embedder_version=self.embedder.version,
                    )
                )
                loaded_event_ids = {event.id for event in direct_events}
                extra_semantic_ids = sorted(semantic_candidate_ids - loaded_event_ids)
                if extra_semantic_ids:
                    for event in visible_events_provider(time_window, extra_semantic_ids):
                        if event.id in loaded_event_ids:
                            continue
                        direct_events.append(event)
                        loaded_event_ids.add(event.id)
                        lexical_scores[event.id] = 0.0
            else:
                semantic_candidate_ids = self.store.event_ids_with_embeddings(self.embedder.version)
                if not semantic_candidate_ids:
                    return []
                direct_events = visible_events_provider(time_window, semantic_candidate_ids)
                if not direct_events:
                    return []
                lexical_scores = {event.id: 0.0 for event in direct_events}
        else:
            semantic_candidate_ids = self.store.event_ids_with_embeddings(self.embedder.version)
            if not semantic_candidate_ids:
                return []
            direct_events = visible_events_provider(time_window, semantic_candidate_ids)
            if not direct_events:
                return []
            lexical_scores = {event.id: 0.0 for event in direct_events}

        semantic_scores = self._semantic_scores(query, direct_events)

        scored_events_by_id: dict[str, ScoredEvent] = {}
        for event in direct_events:
            lexical_score = lexical_scores.get(event.id, 0.0)
            semantic_score = semantic_scores.get(event.id, 0.0)
            if semantic_score > 0:
                direct_score = (
                    (_LEXICAL_WEIGHT * lexical_score if lexical_score > 0 else 0.0)
                    + (_SEMANTIC_WEIGHT * semantic_score if semantic_score > 0 else 0.0)
                )
            else:
                direct_score = lexical_score
            if lexical_score <= 0 and semantic_score <= 0:
                continue
            if direct_score <= 0:
                continue
            scored_events_by_id[event.id] = ScoredEvent(
                event=event,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                direct_score=direct_score,
            )

        if not scored_events_by_id:
            return []

        visible_events_by_id = {event.id: event for event in direct_events}
        causal_candidate_ids = self._causal_candidate_event_ids(
            seed_events=list(scored_events_by_id.values()),
            loaded_event_ids=set(visible_events_by_id),
        )
        if causal_candidate_ids:
            for event in visible_events_provider(time_window, causal_candidate_ids):
                visible_events_by_id.setdefault(event.id, event)
        causal_scores = self._causal_scores(
            seed_events=list(scored_events_by_id.values()),
            visible_events_by_id=visible_events_by_id,
        )
        for event_id, score in causal_scores.items():
            if event_id in scored_events_by_id:
                continue
            event = visible_events_by_id.get(event_id)
            if event is None:
                continue
            scored_events_by_id[event_id] = ScoredEvent(
                event=event,
                lexical_score=0.0,
                semantic_score=0.0,
                direct_score=0.0,
                causal_score=score,
            )

        scored_events = list(scored_events_by_id.values())
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
            matched_axes: set[str] = set()
            if any(item.direct_score > 0 for item in ranked_events):
                matched_axes.add("entity")
            if any(item.semantic_score > 0 for item in ranked_events):
                matched_axes.add("semantic")
            if any(item.causal_score > 0 for item in ranked_events):
                matched_axes.add("causal")
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

    def _known_visible_events(
        self,
        time_window: tuple[datetime, datetime] | None,
        candidate_event_ids: list[str] | None,
    ) -> list[Event]:
        upper_bound = time_window[1] if time_window else utcnow()
        lower_bound = time_window[0] if time_window else None
        events = self.store.visible_events_known(
            to_rfc3339(upper_bound),
            from_recorded_at=to_rfc3339(lower_bound) if lower_bound else None,
            event_ids=candidate_event_ids,
        )
        return self.store.filter_relation_events_live_known(events, to_rfc3339(upper_bound))

    def _valid_visible_events(
        self,
        time_window: tuple[datetime, datetime] | None,
        candidate_event_ids: list[str] | None,
        *,
        relation_window_cache: RelationWindowQueryCache | None = None,
    ) -> list[Event]:
        visible_events = self.store.visible_events_valid(event_ids=candidate_event_ids)
        if time_window is None:
            as_of = utcnow()
            return self.store.filter_relation_events_live_valid_at(
                [event for event in visible_events if covers_valid_time(event, as_of)],
                as_of,
            )

        start_at, end_at = time_window
        return self.store.filter_relation_events_live_valid_in_window(
            [
                event
                for event in visible_events
                if overlaps_valid_time_window(event, start_at, end_at)
            ],
            start_at,
            end_at,
            query_cache=relation_window_cache,
        )

    def _semantic_scores(self, query: str, events: list[Event]) -> dict[str, float]:
        event_embeddings = self.store.event_embeddings_for_ids(
            [event.id for event in events],
            embedder_version=self.embedder.version,
        )
        if not event_embeddings:
            return {}

        scores: dict[str, float] = {}
        query_vector = self._query_embedding(query)
        for event in events:
            event_vector = event_embeddings.get(event.id)
            if event_vector is None:
                continue
            score = max(0.0, cosine_similarity(query_vector, event_vector))
            if score >= _SEMANTIC_MIN_SCORE:
                scores[event.id] = score
        return scores

    def _causal_scores(
        self,
        *,
        seed_events: list[ScoredEvent],
        visible_events_by_id: dict[str, Event],
    ) -> dict[str, float]:
        downstream_by_id: dict[str, list[Event]] = defaultdict(list)
        for event in visible_events_by_id.values():
            if event.caused_by is None:
                continue
            if event.caused_by not in visible_events_by_id:
                continue
            downstream_by_id[event.caused_by].append(event)

        scores: dict[str, float] = defaultdict(float)
        for scored in seed_events:
            if scored.direct_score <= 0:
                continue
            causal_score = scored.direct_score * _CAUSAL_WEIGHT
            if causal_score <= 0:
                continue
            if scored.event.caused_by is not None and scored.event.caused_by in visible_events_by_id:
                scores[scored.event.caused_by] += causal_score
            for event in downstream_by_id.get(scored.event.id, []):
                scores[event.id] += causal_score
        return dict(scores)

    def _causal_candidate_event_ids(
        self,
        *,
        seed_events: list[ScoredEvent],
        loaded_event_ids: set[str],
    ) -> list[str]:
        seed_ids = [scored.event.id for scored in seed_events if scored.direct_score > 0]
        if not seed_ids:
            return []
        upstream_ids = {
            scored.event.caused_by
            for scored in seed_events
            if scored.direct_score > 0 and scored.event.caused_by is not None
        }
        downstream_ids = set(self.store.event_ids_by_caused_by(seed_ids))
        candidate_ids = (upstream_ids | downstream_ids) - loaded_event_ids
        return sorted(candidate_ids)

    def _query_embedding(self, query: str) -> list[float]:
        normalized_query = query.strip().lower()
        cache_key = (self.embedder.version, normalized_query)
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            self._query_embedding_cache.move_to_end(cache_key)
            return cached

        query_embedding = self.embedder.embed_texts([query])
        if len(query_embedding) != 1:
            raise ValueError(f"embedder returned {len(query_embedding)} query embeddings")
        vector = query_embedding[0]
        self._query_embedding_cache[cache_key] = vector
        self._query_embedding_cache.move_to_end(cache_key)
        while len(self._query_embedding_cache) > _QUERY_EMBED_CACHE_SIZE:
            self._query_embedding_cache.popitem(last=False)
        return vector


def _score_event_lexical(event: Event, tokens: list[QueryToken]) -> float:
    if not tokens:
        return 0.0
    haystack = event_search_text(event)
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
