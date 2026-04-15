from __future__ import annotations

import queue
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .canonical import CanonicalWorker, Extractor, NullExtractor
from .errors import ValidationError
from .event_ops import derive_dirty_rows, derive_event_entities, validate_event
from .projector import Projector
from .recovery import RecoveryService
from .storage import EventStore, SegmentedRawLog, WriterLock, open_connection
from .time_utils import ensure_utc, to_rfc3339, utcnow
from .types import Entity, Event, HistoryEntry, QueueItem, RawTurn, TemporalEntityView, TurnAck


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

        self._writer_lock = WriterLock(self.root / ".writer.lock")
        self.conn = None
        self._writer_lock.acquire()
        try:
            self.conn = open_connection(self.db_path)
            self.store = EventStore(self.conn)
            self.raw_log = SegmentedRawLog(self.root / "raw")
            self.projector = Projector(self.store)
            self.canonical_worker = CanonicalWorker(self.store, self.extractor)
            self.queue: queue.Queue[QueueItem] = queue.Queue(maxsize=queue_max_size)
            self.recovery = RecoveryService(
                raw_log=self.raw_log,
                store=self.store,
                projector=self.projector,
                work_queue=self.queue,
                queue_put_timeout=self.queue_put_timeout,
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
        confidence: float | None = None,
        reason: str | None = None,
        time_confidence: str = "unknown",
    ) -> str:
        observed = ensure_utc(observed_at, "observed_at")
        effective_start = ensure_utc(effective_at_start, "effective_at_start") if effective_at_start else None
        effective_end = ensure_utc(effective_at_end, "effective_at_end") if effective_at_end else None
        validate_event(event_type, data)

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
                caused_by=None,
                schema_version=1,
            )
            event_entities = derive_event_entities(event)
            dirty_rows = derive_dirty_rows(event, event_entities)
            self.store.append_event(tx, event)
            self.store.append_event_entities(tx, event.id, event_entities)
            self.store.mark_dirty(tx, dirty_rows)
        return event.id

    def get(self, entity_id: str) -> Entity | None:
        now = utcnow()
        events = self.store.entity_events_visible_at(entity_id, to_rfc3339(now))
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
        events = self.store.entity_events_visible_at(entity_id, to_rfc3339(target))
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

    def known_history(self, entity_id: str, attr: str | None = None) -> list[HistoryEntry]:
        events = self.store.entity_events(entity_id)
        current: dict = {}
        history: list[HistoryEntry] = []
        for event in events:
            if not event.type.startswith("entity.") or event.data["id"] != entity_id:
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
                        basis="known",
                        event_id=event.id,
                    )
                )
                current[key] = new_value
        return history

    def raw_get(self, turn_id: str) -> RawTurn | None:
        return self.raw_log.raw_get(turn_id)

    def raw_recent(self, limit: int = 20) -> list[RawTurn]:
        return self.raw_log.raw_recent(limit=limit)

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
            while self.store.count_dirty_ranges() > 0:
                rebuilt = self.projector.rebuild_dirty()
                if rebuilt == 0 and self.store.count_dirty_ranges() > 0:
                    raise RuntimeError("projection flush made no progress")
            return
        if level == "index":
            raise NotImplementedError("flush(level='index') is planned for semantic indexing")
        raise ValidationError(f"unsupported flush level: {level}")
