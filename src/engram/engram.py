from __future__ import annotations

from contextlib import contextmanager
import queue
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .canonical import CanonicalWorker, Extractor, NullExtractor
from .context_builder import ContextBuilder
from .errors import ValidationError
from .event_ops import derive_cascade_dirty_rows_for_entity_event, derive_dirty_rows, derive_event_entities, validate_event
from .meaning_index import MeaningAnalyzer, MeaningIndexer, NullMeaningAnalyzer
from .projector import Projector
from .recovery import RecoveryService
from .retrieval import RetrievalEngine
from .search_terms import event_search_terms
from .semantic import Embedder, HashEmbedder
from .semantic_index import SemanticIndexer
from .storage import EventStore, SegmentedRawLog, WriterLock, open_connection
from .storage.store import valid_event_sort_key
from .time_utils import ensure_utc, to_rfc3339, utcnow
from .types import (
    Entity,
    Event,
    HistoryEntry,
    ProjectionRebuildResult,
    QueueItem,
    RawTurn,
    RelationEdge,
    SearchResult,
    TemporalEntityView,
    TurnAck,
)
from .types import RelationHistoryEntry


def _safe_user_id(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", user_id).strip("._")
    return safe or "default"


class Engram:
    def __init__(
        self,
        user_id: str = "default",
        path: str | None = None,
        session_id: str | None = None,
        queue_max_size: int = 10000,
        queue_put_timeout: float = 1.0,
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
        meaning_analyzer: MeaningAnalyzer | None = None,
        auto_flush: bool = False,
    ):
        from .retry import RetryPolicy

        base_root = Path(path) if path else Path.home() / ".engram" / "users"
        self.user_id = user_id
        self.safe_user_id = _safe_user_id(user_id)
        self.root = base_root / self.safe_user_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "engram.db"
        self.session_id = session_id
        self.queue_put_timeout = queue_put_timeout
        self.extractor = extractor or NullExtractor()
        self.embedder = embedder or HashEmbedder()
        self.meaning_analyzer = meaning_analyzer or NullMeaningAnalyzer()

        self._writer_lock = WriterLock(self.root / ".writer.lock")
        self.conn = None
        self._background_worker = None
        self._writer_lock.acquire()
        try:
            self.conn = open_connection(self.db_path)
            self.store = EventStore(self.conn, db_path=self.db_path if auto_flush else None)
            self.raw_log = SegmentedRawLog(self.root / "raw")
            self._bind_extractor_runtime_context()
            self.projector = Projector(self.store)
            self.canonical_worker = CanonicalWorker(self.store, self.extractor)
            self.semantic_indexer = SemanticIndexer(self.store, self.embedder)
            self.meaning_indexer = MeaningIndexer(self.store, self.meaning_analyzer)
            self.retrieval = RetrievalEngine(self.store, self.embedder, self.meaning_analyzer)
            self.context_builder = ContextBuilder(self.store, self.raw_log.raw_get)
            self.queue: queue.Queue[QueueItem] = queue.Queue(maxsize=queue_max_size)
            self.recovery = RecoveryService(
                raw_log=self.raw_log,
                store=self.store,
                projector=self.projector,
                semantic_indexer=self.semantic_indexer,
                meaning_indexer=self.meaning_indexer,
                work_queue=self.queue,
                queue_put_timeout=self.queue_put_timeout,
                extractor_version=self.extractor.version,
            )
            self.recovery.catch_up_on_startup()
            if auto_flush:
                from .background import BackgroundWorker

                self._background_worker = BackgroundWorker(
                    work_queue=self.queue,
                    canonical_worker=self.canonical_worker,
                    projector=self.projector,
                    semantic_indexer=self.semantic_indexer,
                    meaning_indexer=self.meaning_indexer,
                    retry_policy=RetryPolicy(),
                )
                self._background_worker.start()
        except BaseException:
            if self._background_worker is not None:
                self._background_worker.stop()
                self._background_worker = None
            if hasattr(self, "store"):
                self.store.close_readers()
            if self.conn is not None:
                self.conn.close()
                self.conn = None
            self._writer_lock.release()
            raise

    def close(self) -> None:
        try:
            if self._background_worker is not None:
                self._background_worker.stop()
                self._background_worker = None
            self.store.close_readers()
            if self.conn is not None:
                self.conn.close()
                self.conn = None
        finally:
            self._writer_lock.release()

    @contextmanager
    def _paused_background_worker(self):
        worker = self._background_worker
        if worker is None:
            yield
            return
        worker.stop()
        try:
            yield
        finally:
            worker.start()

    def _request_background_maintenance(self) -> None:
        if self._background_worker is not None:
            self._background_worker.request_maintenance()

    def _rebuild_projection_dirty(self) -> int:
        return self.projector.rebuild_dirty_until_stable()

    def turn(
        self,
        user: str,
        assistant: str,
        *,
        observed_at=None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> TurnAck:
        observed = ensure_utc(observed_at, "observed_at")
        turn = RawTurn(
            id=str(uuid4()),
            session_id=session_id or self.session_id,
            observed_at=observed,
            user=user,
            assistant=assistant,
            metadata=metadata or {},
        )
        ack = self.raw_log.append(turn)
        item = QueueItem.from_turn(turn)
        try:
            self.queue.put(item, timeout=self.queue_put_timeout)
        except queue.Full:
            return TurnAck(turn_id=ack.turn_id, observed_at=ack.observed_at, durable_at=ack.durable_at, queued=False)
        if self._background_worker is not None:
            self._background_worker.notify()
        return ack

    def append(
        self,
        event_type: str,
        data: dict,
        *,
        observed_at=None,
        effective_at_start=None,
        effective_at_end=None,
        source_role: str = "manual",
        source_turn_id: str | None = None,
        caused_by: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        time_confidence: str = "unknown",
    ) -> str:
        observed = ensure_utc(observed_at, "observed_at")
        effective_start = ensure_utc(effective_at_start, "effective_at_start") if effective_at_start else None
        effective_end = ensure_utc(effective_at_end, "effective_at_end") if effective_at_end else None
        validate_event(event_type, data)

        with self.store.transaction() as tx:
            if caused_by is not None and not self.store.event_exists_in_tx(tx, caused_by):
                raise ValidationError(f"caused_by event not found: {caused_by}")
            recorded_at = utcnow()
            event = Event(
                id=str(uuid4()),
                seq=self.store.next_seq(tx),
                observed_at=observed,
                effective_at_start=effective_start,
                effective_at_end=effective_end,
                recorded_at=recorded_at,
                type=event_type,
                data=data,
                extraction_run_id=None,
                source_turn_id=source_turn_id,
                source_role=source_role,
                confidence=confidence,
                reason=reason,
                time_confidence=time_confidence,
                caused_by=caused_by,
                schema_version=1,
            )
            event_entities = derive_event_entities(event)
            dirty_rows = derive_dirty_rows(event, event_entities)
            dirty_rows.extend(
                derive_cascade_dirty_rows_for_entity_event(
                    event,
                    self.store.related_owner_ids_for_entity(event.data["id"]) if event.type.startswith("entity.") else [],
                )
            )
            self.store.append_event(tx, event)
            self.store.append_event_entities(tx, event.id, event_entities)
            self.store.append_event_search_terms(tx, event.id, event_search_terms(event))
            self.store.mark_dirty(tx, dirty_rows)
        self._request_background_maintenance()
        return event.id

    def get(self, entity_id: str) -> Entity | None:
        now = utcnow()
        events = self.store.entity_events_known_visible_at(entity_id, to_rfc3339(now))
        folded = self.store.fold_entity_events(entity_id, events)
        if folded is None:
            return None
        return Entity(
            id=folded.entity_id,
            type=folded.entity_type,
            attrs=dict(folded.attrs),
            created_recorded_at=folded.created_recorded_at,
            updated_recorded_at=folded.updated_recorded_at,
        )

    def get_known_at(self, entity_id: str, at) -> TemporalEntityView | None:
        target = ensure_utc(at, "at")
        events = self.store.entity_events_known_visible_at(entity_id, to_rfc3339(target))
        folded = self.store.fold_entity_events(entity_id, events)
        if folded is None:
            return None
        return TemporalEntityView(
            entity_id=folded.entity_id,
            entity_type=folded.entity_type,
            attrs=dict(folded.attrs),
            unknown_attrs=[],
            supporting_event_ids=list(folded.supporting_event_ids),
            basis="known",
            as_of=target,
        )

    def get_valid_at(self, entity_id: str, at) -> TemporalEntityView | None:
        target = ensure_utc(at, "at")
        folded = self.store.fold_entity_events_valid_at(
            entity_id,
            target,
            events=self.store.entity_events_valid_visible(entity_id),
        )
        if folded is None:
            return None
        return TemporalEntityView(
            entity_id=folded.entity_id,
            entity_type=folded.entity_type,
            attrs=dict(folded.attrs),
            unknown_attrs=list(folded.unknown_attrs),
            supporting_event_ids=list(folded.supporting_event_ids),
            basis="valid",
            as_of=target,
        )

    def known_history(self, entity_id: str, attr: str | None = None) -> list[HistoryEntry]:
        return self._build_history(
            entity_id=entity_id,
            attr=attr,
            events=self.store.entity_events_known_visible_at(entity_id, to_rfc3339(utcnow())),
            basis="known",
            skip_unknown_effective=False,
        )

    def valid_history(self, entity_id: str, attr: str | None = None) -> list[HistoryEntry]:
        return self._build_history(
            entity_id=entity_id,
            attr=attr,
            events=sorted(self.store.entity_events_valid_visible(entity_id), key=valid_event_sort_key),
            basis="valid",
            skip_unknown_effective=True,
        )

    def get_relations(
        self,
        entity_id: str,
        *,
        time_mode: Literal["known", "valid"] = "known",
        at: datetime | None = None,
        time_window: tuple[datetime, datetime] | None = None,
    ) -> list[RelationEdge]:
        if time_mode == "known":
            if time_window is not None:
                raise ValidationError("time_window is not supported for known relation reads")
            target = utcnow() if at is None else ensure_utc(at, "at")
            return self.store.relation_edges_known_at(entity_id, to_rfc3339(target))

        if time_mode == "valid":
            if at is not None and time_window is not None:
                raise ValidationError("valid relation reads accept either at or time_window, not both")
            if time_window is not None:
                start = ensure_utc(time_window[0], "time_window[0]")
                end = ensure_utc(time_window[1], "time_window[1]")
                if start >= end:
                    raise ValidationError("time_window[0] must be before time_window[1] for valid relation reads")
                return self.store.relation_edges_valid_in_window(entity_id, start, end)
            target = utcnow() if at is None else ensure_utc(at, "at")
            return self.store.relation_edges_valid_at(entity_id, target)

        raise ValidationError(f"unsupported time_mode: {time_mode}")

    def relation_history(
        self,
        entity_id: str,
        *,
        relation_type: str | None = None,
        other_entity_id: str | None = None,
        time_mode: Literal["known", "valid"] = "known",
    ) -> list[RelationHistoryEntry]:
        if time_mode == "known":
            events = self.store.relation_events_known_visible_at(entity_id, to_rfc3339(utcnow()))
            return self._build_relation_history(
                entity_id=entity_id,
                relation_type=relation_type,
                other_entity_id=other_entity_id,
                events=events,
                basis="known",
            )

        if time_mode == "valid":
            events = sorted(self.store.relation_events_valid_visible(entity_id), key=valid_event_sort_key)
            return self._build_relation_history(
                entity_id=entity_id,
                relation_type=relation_type,
                other_entity_id=other_entity_id,
                events=events,
                basis="valid",
            )

        raise ValidationError(f"unsupported time_mode: {time_mode}")

    def reprocess(
        self,
        *,
        from_turn_id: str | None = None,
        to_turn_id: str | None = None,
        extractor_version: str | None = None,
    ) -> int:
        if extractor_version is not None and extractor_version != self.extractor.version:
            raise ValidationError(
                f"extractor_version must match current extractor version {self.extractor.version}"
            )
        try:
            turns = self.raw_log.raw_range(from_turn_id=from_turn_id, to_turn_id=to_turn_id)
        except KeyError as exc:
            raise ValidationError(f"turn_id not found: {exc.args[0]}") from exc
        except ValueError as exc:
            if str(exc) == "from_turn_id_after_to_turn_id":
                raise ValidationError(f"from_turn_id {from_turn_id} is after to_turn_id {to_turn_id}") from exc
            raise

        background_was_running = self._background_worker is not None
        with self._paused_background_worker():
            count = 0
            for turn in turns:
                self.canonical_worker.process(QueueItem.from_turn(turn), force=True)
                count += 1
        if background_was_running and count > 0:
            self._request_background_maintenance()
        return count

    def rebuild_projection(
        self,
        *,
        owner_id: str | None = None,
        mode: Literal["dirty", "full"] = "dirty",
    ) -> ProjectionRebuildResult:
        with self._paused_background_worker():
            dirty_owner_count_before = len(self.store.dirty_owner_ids())

            if mode == "dirty":
                if owner_id is None:
                    rebuilt_owner_count = self._rebuild_projection_dirty()
                    scope: Literal["dirty", "owner", "full"] = "dirty"
                    target_owner_id = None
                else:
                    current_relation_neighbors = [
                        edge.other_entity_id
                        for edge in self.projector.current_relation_snapshot().get(owner_id, ())
                    ]
                    canonical_relation_neighbors = [
                        edge.other_entity_id for edge in self.store.materialize_current_relations(owner_id)
                    ]
                    rebuilt_owner_count = self.projector.rebuild_owner(
                        owner_id,
                        related_owner_ids=current_relation_neighbors + canonical_relation_neighbors,
                    )
                    scope = "owner"
                    target_owner_id = owner_id
            elif mode == "full":
                if owner_id is not None:
                    raise ValidationError("owner_id is not supported when mode='full'")
                rebuilt_owner_count = self.projector.rebuild_all()
                scope = "full"
                target_owner_id = None
            else:
                raise ValidationError(f"unsupported rebuild mode: {mode}")

            dirty_owner_count_after = len(self.store.dirty_owner_ids())
            return ProjectionRebuildResult(
                scope=scope,
                target_owner_id=target_owner_id,
                rebuilt_owner_count=rebuilt_owner_count,
                dirty_owner_count_before=dirty_owner_count_before,
                dirty_owner_count_after=dirty_owner_count_after,
            )

    def _build_history(
        self,
        *,
        entity_id: str,
        attr: str | None,
        events: list[Event],
        basis: Literal["known", "valid"],
        skip_unknown_effective: bool,
    ) -> list[HistoryEntry]:
        current: dict = {}
        history: list[HistoryEntry] = []
        for event in events:
            if not event.type.startswith("entity.") or event.data["id"] != entity_id:
                continue
            if skip_unknown_effective and event.effective_at_start is None:
                continue
            if event.type == "entity.delete":
                current.clear()
                continue
            attrs = event.data["attrs"] if event.type in {"entity.create", "entity.update"} else {}
            for key, new_value in attrs.items():
                if attr and key != attr:
                    continue
                old_value = current.get(key)
                if old_value == new_value:
                    continue
                history.append(
                    HistoryEntry(
                        entity_id=entity_id,
                        attr=key,
                        old_value=old_value,
                        new_value=new_value,
                        observed_at=event.observed_at,
                        effective_at_start=event.effective_at_start,
                        effective_at_end=event.effective_at_end,
                        recorded_at=event.recorded_at,
                        reason=event.reason,
                        confidence=event.confidence,
                        basis=basis,
                        event_id=event.id,
                    )
                )
                current[key] = new_value
        return history

    def _build_relation_history(
        self,
        *,
        entity_id: str,
        relation_type: str | None,
        other_entity_id: str | None,
        events: list[Event],
        basis: Literal["known", "valid"],
    ) -> list[RelationHistoryEntry]:
        history: list[RelationHistoryEntry] = []
        for event in events:
            if not event.type.startswith("relation."):
                continue
            source = str(event.data["source"])
            target = str(event.data["target"])
            if entity_id == source:
                direction: Literal["outgoing", "incoming"] = "outgoing"
                other_id = target
            elif entity_id == target:
                direction = "incoming"
                other_id = source
            else:
                continue
            current_relation_type = str(event.data["type"])
            if relation_type is not None and current_relation_type != relation_type:
                continue
            if other_entity_id is not None and other_id != other_entity_id:
                continue
            history.append(
                RelationHistoryEntry(
                    entity_id=entity_id,
                    other_entity_id=other_id,
                    relation_type=current_relation_type,
                    direction=direction,
                    action=event.type.split(".", 1)[1],  # type: ignore[arg-type]
                    attrs=dict(event.data.get("attrs", {})),
                    observed_at=event.observed_at,
                    effective_at_start=event.effective_at_start,
                    effective_at_end=event.effective_at_end,
                    recorded_at=event.recorded_at,
                    reason=event.reason,
                    confidence=event.confidence,
                    basis=basis,
                    event_id=event.id,
                )
            )
        return history

    def raw_get(self, turn_id: str) -> RawTurn | None:
        return self.raw_log.raw_get(turn_id)

    def raw_recent(self, limit: int = 20) -> list[RawTurn]:
        return self.raw_log.raw_recent(limit=limit)

    def _bind_extractor_runtime_context(self) -> None:
        bind = getattr(self.extractor, "bind_runtime_context", None)
        if callable(bind):
            bind(
                safe_user_id=self.safe_user_id,
                recent_turns_provider=self._recent_turns_for_extractor,
            )

    def _recent_turns_for_extractor(self, item: QueueItem, limit: int) -> list[RawTurn]:
        if item.session_id is not None:
            recent = self.raw_log.raw_recent_for_session(item.session_id, limit=max(limit + 1, 1))
        else:
            recent = self.raw_recent(limit=max(limit * 4, limit))
        filtered: list[RawTurn] = []
        for turn in recent:
            if turn.id == item.turn_id:
                continue
            filtered.append(turn)
            if len(filtered) == limit:
                break
        return list(reversed(filtered))

    def search(
        self,
        query: str,
        *,
        time_mode: Literal["known", "valid"] = "known",
        time_window=None,
        k: int = 20,
    ) -> list[SearchResult]:
        if time_mode == "known":
            return self.retrieval.search_known(query, k=k, time_window=time_window)
        if time_mode == "valid":
            return self.retrieval.search_valid(query, k=k, time_window=time_window)
        raise ValidationError(f"unsupported time_mode: {time_mode}")

    def context(
        self,
        query: str,
        *,
        time_mode: Literal["known", "valid"] = "known",
        time_window=None,
        max_tokens: int = 2000,
        include_history: bool = True,
        include_raw: bool = False,
    ) -> str:
        as_of = time_window[1] if time_window else utcnow()
        relation_window_cache = None
        if time_mode == "valid" and time_window is not None:
            relation_window_cache = self.store.build_relation_window_query_cache(
                ensure_utc(time_window[0], "time_window[0]"),
                ensure_utc(time_window[1], "time_window[1]"),
            )
            results = self.retrieval.search_valid(
                query,
                k=5,
                time_window=time_window,
                relation_window_cache=relation_window_cache,
            )
        else:
            results = self.search(query, time_mode=time_mode, time_window=time_window, k=5)
        entity_ids = [result.entity_id for result in results]
        if time_mode == "known":
            return self.context_builder.build_known(
                query=query,
                results=results,
                as_of=as_of,
                max_tokens=max_tokens,
                include_history=include_history,
                include_raw=include_raw,
                views_by_entity=self._get_known_views_at_many(entity_ids, as_of),
                relations_by_entity=self._get_known_relations_at_many(entity_ids, as_of),
            )
        if time_mode == "valid":
            valid_at = ensure_utc(as_of, "as_of")
            if time_window is not None:
                relations_by_entity = self._get_valid_relations_in_window_many(
                    entity_ids,
                    time_window[0],
                    time_window[1],
                    relation_window_cache=relation_window_cache,
                )
            else:
                relations_by_entity = self._get_valid_relations_at_many(entity_ids, valid_at)
            return self.context_builder.build_valid(
                query=query,
                results=results,
                as_of=as_of,
                time_window=time_window,
                max_tokens=max_tokens,
                include_history=include_history,
                include_raw=include_raw,
                views_by_entity=self._get_valid_views_at_many(entity_ids, valid_at),
                relations_by_entity=relations_by_entity,
            )
        raise ValidationError(f"unsupported time_mode: {time_mode}")

    def flush(
        self,
        level: Literal["raw", "canonical", "projection", "snapshot", "index", "all"] = "projection",
    ) -> None:
        if level == "raw":
            return
        background_was_running = self._background_worker is not None
        with self._paused_background_worker():
            processed_canonical = self._flush_internal(level)
        if background_was_running and processed_canonical:
            self._request_background_maintenance()

    def _flush_internal(
        self,
        level: Literal["raw", "canonical", "projection", "snapshot", "index", "all"],
    ) -> bool:
        if level == "raw":
            return False
        if level == "canonical":
            processed = 0
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.canonical_worker.process(item)
                finally:
                    self.queue.task_done()
                processed += 1
            return processed > 0
        if level == "projection":
            self._rebuild_projection_dirty()
            return False
        if level == "snapshot":
            self.projector.save_snapshot()
            return False
        if level == "index":
            self.semantic_indexer.index_missing()
            self.meaning_indexer.index_missing()
            return False
        if level == "all":
            processed_canonical = self._flush_internal("canonical")
            self._flush_internal("projection")
            self._flush_internal("snapshot")
            self._flush_internal("index")
            return processed_canonical
        raise ValidationError(f"unsupported flush level: {level}")

    def _get_known_relations_at(self, entity_id: str, at) -> list[RelationEdge]:
        target = ensure_utc(at, "at")
        return self.store.relation_edges_known_at(entity_id, to_rfc3339(target))

    def _get_known_views_at_many(self, entity_ids: list[str], at) -> dict[str, TemporalEntityView | None]:
        target = ensure_utc(at, "at")
        folded_by_entity = self.store.fold_entities_known_at(entity_ids, to_rfc3339(target))
        views: dict[str, TemporalEntityView | None] = {}
        for entity_id in entity_ids:
            folded = folded_by_entity.get(entity_id)
            if folded is None:
                views[entity_id] = None
                continue
            views[entity_id] = TemporalEntityView(
                entity_id=folded.entity_id,
                entity_type=folded.entity_type,
                attrs=dict(folded.attrs),
                unknown_attrs=[],
                supporting_event_ids=list(folded.supporting_event_ids),
                basis="known",
                as_of=target,
            )
        return views

    def _get_known_relations_at_many(self, entity_ids: list[str], at) -> dict[str, list[RelationEdge]]:
        target = ensure_utc(at, "at")
        return self.store.relation_edges_known_at_many(entity_ids, to_rfc3339(target))

    def _get_valid_relations_at(self, entity_id: str, at) -> list[RelationEdge]:
        target = ensure_utc(at, "at")
        return self.store.relation_edges_valid_at(entity_id, target)

    def _get_valid_views_at_many(self, entity_ids: list[str], at) -> dict[str, TemporalEntityView | None]:
        target = ensure_utc(at, "at")
        folded_by_entity = self.store.fold_entities_valid_at(entity_ids, target)
        views: dict[str, TemporalEntityView | None] = {}
        for entity_id in entity_ids:
            folded = folded_by_entity.get(entity_id)
            if folded is None:
                views[entity_id] = None
                continue
            views[entity_id] = TemporalEntityView(
                entity_id=folded.entity_id,
                entity_type=folded.entity_type,
                attrs=dict(folded.attrs),
                unknown_attrs=list(folded.unknown_attrs),
                supporting_event_ids=list(folded.supporting_event_ids),
                basis="valid",
                as_of=target,
            )
        return views

    def _get_valid_relations_at_many(self, entity_ids: list[str], at) -> dict[str, list[RelationEdge]]:
        target = ensure_utc(at, "at")
        return self.store.relation_edges_valid_at_many(entity_ids, target)

    def _get_valid_relations_in_window(
        self,
        entity_id: str,
        start_at,
        end_at,
    ) -> list[RelationEdge]:
        start = ensure_utc(start_at, "start_at")
        end = ensure_utc(end_at, "end_at")
        if start >= end:
            return []
        return self.store.relation_edges_valid_in_window(entity_id, start, end)

    def _get_valid_relations_in_window_many(
        self,
        entity_ids: list[str],
        start_at,
        end_at,
        *,
        relation_window_cache=None,
    ) -> dict[str, list[RelationEdge]]:
        start = ensure_utc(start_at, "start_at")
        end = ensure_utc(end_at, "end_at")
        if start >= end:
            return {entity_id: [] for entity_id in entity_ids}
        return self.store.relation_edges_valid_in_window_many(
            entity_ids,
            start,
            end,
            query_cache=relation_window_cache,
        )
