from __future__ import annotations

import queue
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .canonical import CanonicalWorker, Extractor, NullExtractor
from .context_builder import ContextBuilder
from .errors import ValidationError
from .event_ops import derive_cascade_dirty_rows_for_entity_event, derive_dirty_rows, derive_event_entities, validate_event
from .projector import Projector
from .recovery import RecoveryService
from .retrieval import RetrievalEngine
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
    ):
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

        self._writer_lock = WriterLock(self.root / ".writer.lock")
        self.conn = None
        self._writer_lock.acquire()
        try:
            self.conn = open_connection(self.db_path)
            self.store = EventStore(self.conn)
            self.raw_log = SegmentedRawLog(self.root / "raw")
            self.projector = Projector(self.store)
            self.canonical_worker = CanonicalWorker(self.store, self.extractor)
            self.semantic_indexer = SemanticIndexer(self.store, self.embedder)
            self.retrieval = RetrievalEngine(self.store, self.embedder)
            self.context_builder = ContextBuilder(self.store, self.raw_log.raw_get)
            self.queue: queue.Queue[QueueItem] = queue.Queue(maxsize=queue_max_size)
            self.recovery = RecoveryService(
                raw_log=self.raw_log,
                store=self.store,
                projector=self.projector,
                work_queue=self.queue,
                queue_put_timeout=self.queue_put_timeout,
                extractor_version=self.extractor.version,
            )
            self.recovery.catch_up_on_startup()
        except BaseException:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
            self._writer_lock.release()
            raise

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
        finally:
            self._writer_lock.release()

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
        if caused_by is not None and not self.store.event_exists(caused_by):
            raise ValidationError(f"caused_by event not found: {caused_by}")

        with self.store.transaction() as tx:
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
            self.store.mark_dirty(tx, dirty_rows)
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

        count = 0
        for turn in turns:
            self.canonical_worker.process(QueueItem.from_turn(turn), force=True)
            count += 1
        return count

    def rebuild_projection(
        self,
        *,
        owner_id: str | None = None,
        mode: Literal["dirty", "full"] = "dirty",
    ) -> ProjectionRebuildResult:
        dirty_owner_count_before = len(self.store.dirty_owner_ids())

        if mode == "dirty":
            if owner_id is None:
                rebuilt_owner_count = self.projector.rebuild_dirty_until_stable()
                scope: Literal["dirty", "owner", "full"] = "dirty"
                target_owner_id = None
            else:
                self.projector.rebuild_owner(owner_id)
                rebuilt_owner_count = 1
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
        results = self.search(query, time_mode=time_mode, time_window=time_window, k=5)
        if time_mode == "known":
            return self.context_builder.build_known(
                query=query,
                results=results,
                as_of=as_of,
                max_tokens=max_tokens,
                include_history=include_history,
                include_raw=include_raw,
                get_known_at=self.get_known_at,
                get_known_relations_at=self._get_known_relations_at,
            )
        if time_mode == "valid":
            return self.context_builder.build_valid(
                query=query,
                results=results,
                as_of=as_of,
                time_window=time_window,
                max_tokens=max_tokens,
                include_history=include_history,
                include_raw=include_raw,
                get_valid_at=self.get_valid_at,
                get_valid_relations_at=self._get_valid_relations_at,
                get_valid_relations_in_window=self._get_valid_relations_in_window,
            )
        raise ValidationError(f"unsupported time_mode: {time_mode}")

    def flush(
        self,
        level: Literal["raw", "canonical", "projection", "index"] = "projection",
    ) -> None:
        if level == "raw":
            return
        if level == "canonical":
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    self.canonical_worker.process(item)
                finally:
                    self.queue.task_done()
        if level == "projection":
            self.rebuild_projection(mode="dirty")
            return
        if level == "index":
            self.semantic_indexer.index_missing()
            return
        raise ValidationError(f"unsupported flush level: {level}")

    def _get_known_relations_at(self, entity_id: str, at) -> list[RelationEdge]:
        target = ensure_utc(at, "at")
        return self.store.relation_edges_known_at(entity_id, to_rfc3339(target))

    def _get_valid_relations_at(self, entity_id: str, at) -> list[RelationEdge]:
        target = ensure_utc(at, "at")
        return self.store.relation_edges_valid_at(entity_id, target)

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
