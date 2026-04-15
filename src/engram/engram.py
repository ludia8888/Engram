from __future__ import annotations

import queue
import re
from pathlib import Path
from uuid import uuid4

from .errors import QueueFullError, ValidationError
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
    ):
        base_root = Path(path) if path else Path.home() / ".engram" / "users"
        self.user_id = user_id
        self.safe_user_id = _safe_user_id(user_id)
        self.root = base_root / self.safe_user_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "engram.db"
        self.session_id = session_id
        self.queue_put_timeout = queue_put_timeout

        self._writer_lock = WriterLock(self.root / ".writer.lock")
        self._writer_lock.acquire()

        self.conn = open_connection(self.db_path)
        self.store = EventStore(self.conn)
        self.raw_log = SegmentedRawLog(self.root / "raw")
        self.queue: queue.Queue[QueueItem] = queue.Queue(maxsize=queue_max_size)

    def close(self) -> None:
        try:
            self.conn.close()
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
        except queue.Full as exc:
            raise QueueFullError("turn queue is full") from exc
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
        self._validate_event(event_type, data)

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
            event_entities = self._derive_event_entities(event)
            dirty_rows = self._derive_dirty_rows(event, event_entities)
            self.store.append_event(tx, event)
            self.store.append_event_entities(tx, event.id, event_entities)
            self.store.mark_dirty(tx, dirty_rows)
        return event.id

    def get(self, entity_id: str) -> Entity | None:
        view = self.get_known_at(entity_id, utcnow())
        if view is None:
            return None
        history = self.store.entity_events_visible_at(entity_id, to_rfc3339(utcnow()))
        created_at = history[0].recorded_at if history else utcnow()
        updated_at = history[-1].recorded_at if history else created_at
        return Entity(
            id=view.entity_id,
            type=view.entity_type,
            attrs=dict(view.attrs),
            created_recorded_at=created_at,
            updated_recorded_at=updated_at,
        )

    def get_known_at(self, entity_id: str, at) -> TemporalEntityView | None:
        target = ensure_utc(at, "at")
        events = self.store.entity_events_visible_at(entity_id, to_rfc3339(target))
        state = None
        attrs: dict = {}
        entity_type = "unknown"
        supporting_event_ids: list[str] = []
        deleted = False
        for event in events:
            if not event.type.startswith("entity."):
                continue
            if event.data["id"] != entity_id:
                continue
            supporting_event_ids.append(event.id)
            deleted = False
            if event.type == "entity.create":
                entity_type = event.data["type"]
                attrs = dict(event.data["attrs"])
                state = True
            elif event.type == "entity.update":
                attrs.update(event.data["attrs"])
                state = True
            elif event.type == "entity.delete":
                deleted = True
                state = None
                attrs = {}
        if state is None or deleted:
            return None
        return TemporalEntityView(
            entity_id=entity_id,
            entity_type=entity_type,
            attrs=dict(attrs),
            unknown_attrs=[],
            supporting_event_ids=supporting_event_ids,
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

    def _validate_event(self, event_type: str, data: dict) -> None:
        if event_type == "entity.create":
            if not isinstance(data.get("id"), str) or not isinstance(data.get("type"), str):
                raise ValidationError("entity.create requires string id and type")
            if not isinstance(data.get("attrs"), dict):
                raise ValidationError("entity.create requires attrs dict")
            return
        if event_type == "entity.update":
            if not isinstance(data.get("id"), str):
                raise ValidationError("entity.update requires string id")
            if not isinstance(data.get("attrs"), dict):
                raise ValidationError("entity.update requires attrs dict")
            return
        if event_type == "entity.delete":
            if not isinstance(data.get("id"), str):
                raise ValidationError("entity.delete requires string id")
            return
        if event_type in {"relation.create", "relation.update"}:
            if not all(isinstance(data.get(key), str) for key in ("source", "target", "type")):
                raise ValidationError(f"{event_type} requires source/target/type strings")
            if "attrs" in data and not isinstance(data["attrs"], dict):
                raise ValidationError(f"{event_type} attrs must be dict when present")
            data.setdefault("attrs", {})
            return
        if event_type == "relation.delete":
            if not all(isinstance(data.get(key), str) for key in ("source", "target", "type")):
                raise ValidationError("relation.delete requires source/target/type strings")
            return
        raise ValidationError(f"unsupported event type: {event_type}")

    def _derive_event_entities(self, event: Event) -> list[tuple[str, str]]:
        if event.type.startswith("entity."):
            return [(event.data["id"], "subject")]
        return [
            (event.data["source"], "source"),
            (event.data["target"], "target"),
        ]

    def _derive_dirty_rows(
        self,
        event: Event,
        event_entities: list[tuple[str, str]],
    ) -> list[tuple[str, str | None, str | None, str, str]]:
        created_at = to_rfc3339(utcnow())
        from_recorded_at = to_rfc3339(event.recorded_at)
        from_effective_at = (
            to_rfc3339(event.effective_at_start) if event.effective_at_start else None
        )
        rows = []
        for entity_id, _role in event_entities:
            rows.append(
                (
                    str(uuid4()),
                    entity_id,
                    from_recorded_at,
                    from_effective_at,
                    f"{event.type}:{event.id}",
                    created_at,
                )
            )
        return rows

