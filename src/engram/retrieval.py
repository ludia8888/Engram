from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

from .meaning_index import (
    MeaningAnalyzer,
    NullMeaningAnalyzer,
    deserialize_query_meaning_plan,
    normalize_query_for_meaning_cache,
    serialize_query_meaning_plan,
)
from .search_terms import (
    QueryToken,
    event_search_text,
    query_candidate_terms,
    query_token_term_groups,
    query_tokens,
)
from .semantic import Embedder, cosine_similarity
from .storage.store import (
    EventStore,
    RelationWindowQueryCache,
    covers_valid_time,
    overlaps_valid_time_window,
    valid_event_sort_key,
)
from .time_utils import to_rfc3339, utcnow
from .types import Event, MeaningUnit, QueryMeaningPlan, SearchResult


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
_MEANING_KIND_WEIGHTS: dict[str, float] = {
    "protected_phrase": 2.0,
    "canonical_key": 1.7,
    "alias": 1.4,
    "facet": 0.8,
}


class RetrievalEngine:
    def __init__(
        self,
        store: EventStore,
        embedder: Embedder,
        meaning_analyzer: MeaningAnalyzer | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.meaning_analyzer = meaning_analyzer or NullMeaningAnalyzer()
        self._fallback_meaning_analyzer = NullMeaningAnalyzer()
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
        meaning_plan = self._query_meaning_plan(query)
        meaning_direct = self._meaning_direct_matches(
            meaning_plan=meaning_plan,
            time_window=time_window,
            visible_events_provider=visible_events_provider,
        )
        if meaning_direct is not None:
            direct_events, lexical_scores = meaning_direct
        else:
            direct_events, lexical_scores = self._fallback_direct_matches(
                query=query,
                tokens=tokens,
                time_mode=time_mode,
                time_window=time_window,
                visible_events_provider=visible_events_provider,
                fallback_terms=meaning_plan.fallback_terms,
            )
            if not direct_events:
                semantic_candidate_ids = self.store.event_ids_with_embeddings(self.embedder.version)
                if not semantic_candidate_ids:
                    return []
                direct_events = visible_events_provider(time_window, semantic_candidate_ids)
                if not direct_events:
                    return []
                lexical_scores = {event.id: 0.0 for event in direct_events}

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

    def _known_visible_lexical_candidate_hits(
        self,
        query: str,
        time_window: tuple[datetime, datetime] | None,
    ) -> dict[str, int]:
        upper_bound = time_window[1] if time_window else utcnow()
        lower_bound = time_window[0] if time_window else None
        return self.store.known_visible_event_token_hits(
            to_rfc3339(upper_bound),
            query_token_term_groups(query),
            from_recorded_at=to_rfc3339(lower_bound) if lower_bound else None,
        )

    def _known_lexical_scores(
        self,
        events: list[Event],
        *,
        tokens: list[QueryToken],
        lexical_token_hits: dict[str, int],
    ) -> dict[str, float]:
        if not tokens:
            return {}
        phrase = " ".join(token.raw for token in tokens)
        all_tokens_matched = len(tokens)
        scores: dict[str, float] = {}
        for event in events:
            matched_count = lexical_token_hits.get(event.id, 0)
            if matched_count <= 0:
                scores[event.id] = 0.0
                continue
            score = matched_count / max(len(tokens), 1)
            if phrase and matched_count >= all_tokens_matched and phrase in event_search_text(event):
                score += 0.25
            scores[event.id] = score
        return scores

    def _query_meaning_plan(self, query: str) -> QueryMeaningPlan:
        normalized_query = normalize_query_for_meaning_cache(query)
        if not normalized_query:
            return QueryMeaningPlan(units=[], fallback_terms=[], planner_confidence=None)

        cached = self.store.load_query_meaning_cache(
            normalized_query,
            self.meaning_analyzer.version,
        )
        if cached is not None:
            return deserialize_query_meaning_plan(cached)

        try:
            plan = self.meaning_analyzer.plan_query(query)
        except Exception:
            plan = self._fallback_meaning_analyzer.plan_query(query)
        if not plan.fallback_terms:
            plan = QueryMeaningPlan(
                units=list(plan.units),
                fallback_terms=query_candidate_terms(query),
                planner_confidence=plan.planner_confidence,
            )

        with self.store.transaction() as tx:
            self.store.save_query_meaning_cache(
                tx,
                normalized_query=normalized_query,
                analyzer_version=self.meaning_analyzer.version,
                payload=serialize_query_meaning_plan(plan),
                cached_at=to_rfc3339(utcnow()),
            )
        return plan

    def _meaning_direct_matches(
        self,
        *,
        meaning_plan: QueryMeaningPlan,
        time_window: tuple[datetime, datetime] | None,
        visible_events_provider: VisibleEventsProvider,
    ) -> tuple[list[Event], dict[str, float]] | None:
        lookups = self._meaning_lookups(meaning_plan.units)
        if not lookups:
            return None

        matches = self.store.event_meaning_matches(self.meaning_analyzer.version, lookups)
        if not matches:
            return None

        direct_events = visible_events_provider(time_window, sorted(matches))
        if not direct_events:
            return None

        lexical_scores = {
            event.id: self._meaning_score(matches.get(event.id, ()))
            for event in direct_events
        }
        if not any(score > 0 for score in lexical_scores.values()):
            return None
        return direct_events, lexical_scores

    def _fallback_direct_matches(
        self,
        *,
        query: str,
        tokens: list[QueryToken],
        time_mode: Literal["known", "valid"],
        time_window: tuple[datetime, datetime] | None,
        visible_events_provider: VisibleEventsProvider,
        fallback_terms: list[str],
    ) -> tuple[list[Event], dict[str, float]]:
        direct_events: list[Event]
        lexical_scores: dict[str, float]
        lexical_token_hits = self._known_visible_lexical_candidate_hits(query, time_window) if time_mode == "known" else None
        lexical_candidate_ids = (
            set(lexical_token_hits)
            if lexical_token_hits is not None
            else set(self.store.candidate_event_ids_for_search_terms(fallback_terms))
        )
        if lexical_candidate_ids:
            if lexical_token_hits is not None:
                upper_bound = to_rfc3339(time_window[1] if time_window else utcnow())
                direct_events = self.store.filter_relation_events_live_known(
                    self.store.events_by_ids(sorted(lexical_candidate_ids)),
                    upper_bound,
                )
                if not direct_events:
                    return [], {}
                lexical_scores = self._known_lexical_scores(
                    direct_events,
                    tokens=tokens,
                    lexical_token_hits=lexical_token_hits,
                )
            else:
                direct_events = visible_events_provider(time_window, sorted(lexical_candidate_ids))
                if not direct_events:
                    return [], {}
                lexical_scores = {
                    event.id: (
                        _score_event_lexical(event, tokens)
                        if event.id in lexical_candidate_ids
                        else 0.0
                    )
                    for event in direct_events
                }
            return direct_events, lexical_scores
        return [], {}

    def _meaning_lookups(
        self,
        units: list[MeaningUnit],
    ) -> list[tuple[str, str, str]]:
        lookups: list[tuple[str, str, str]] = []
        for unit in units:
            if unit.kind == "fallback_term":
                continue
            lookups.append((unit.kind, unit.key or "", unit.normalized_value))
        return list(dict.fromkeys(lookups))

    def _meaning_score(
        self,
        matches: list[tuple[str, str, str, float | None]] | tuple[tuple[str, str, str, float | None], ...],
    ) -> float:
        score = 0.0
        seen: set[tuple[str, str, str]] = set()
        for unit_kind, unit_key, normalized_value, confidence in matches:
            match_key = (unit_kind, unit_key, normalized_value)
            if match_key in seen:
                continue
            seen.add(match_key)
            score += _MEANING_KIND_WEIGHTS.get(unit_kind, 0.0) * (confidence if confidence is not None else 1.0)
        return score

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

        query_embedding = self.embedder.embed_texts([normalized_query])
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
