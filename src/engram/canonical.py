from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .errors import ValidationError
from .event_ops import (
    derive_cascade_dirty_rows_for_entity_event,
    derive_dirty_rows,
    derive_event_entities,
    validate_event,
)
from .storage.store import DirtyRangeRow, EventStore
from .time_utils import to_rfc3339, utcnow
from .types import Event, ExtractedEvent, ExtractionRun, QueueItem


class Extractor(Protocol):
    version: str

    def extract(self, item: QueueItem) -> list[ExtractedEvent]: ...


@dataclass(slots=True)
class NullExtractor:
    version: str = "noop-v1"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        return []


class CanonicalWorker:
    def __init__(self, store: EventStore, extractor: Extractor):
        self.store = store
        self.extractor = extractor

    def process(self, item: QueueItem, *, force: bool = False) -> bool:
        if not force and self.store.has_successful_extraction_run(item.turn_id, self.extractor.version):
            return False

        processed_at = utcnow()
        try:
            drafts = self.extractor.extract(item)
        except Exception as exc:
            self._record_failed_run(item, processed_at, str(exc))
            raise

        try:
            for draft in drafts:
                validate_event(draft.type, draft.data)
                if draft.caused_by is not None and not self.store.event_exists(draft.caused_by):
                    raise ValidationError(f"caused_by event not found: {draft.caused_by}")
            run_id = str(uuid4())
            active_runs = self.store.active_successful_runs_for_turn(item.turn_id)
            superseded_run_ids = [run.id for run in active_runs]
            superseded_owner_ids = self.store.entity_owner_ids_for_runs(superseded_run_ids)
            with self.store.transaction() as tx:
                if superseded_run_ids:
                    self.store.supersede_runs(
                        tx,
                        old_run_ids=superseded_run_ids,
                        new_run_id=run_id,
                        superseded_at=to_rfc3339(processed_at),
                    )

                run = ExtractionRun(
                    id=run_id,
                    source_turn_id=item.turn_id,
                    extractor_version=self.extractor.version,
                    observed_at=item.observed_at,
                    processed_at=processed_at,
                    status="SUCCEEDED",
                    error=None,
                    event_count=len(drafts),
                    superseded_at=None,
                    projection_version=None,
                )
                self.store.append_extraction_run(tx, run)

                next_seq = self.store.next_seq(tx)
                dirty_rows: list[DirtyRangeRow] = []
                for offset, draft in enumerate(drafts):
                    event = Event(
                        id=str(uuid4()),
                        seq=next_seq + offset,
                        observed_at=item.observed_at,
                        effective_at_start=draft.effective_at_start,
                        effective_at_end=draft.effective_at_end,
                        recorded_at=processed_at,
                        type=draft.type,
                        data=draft.data,
                        extraction_run_id=run_id,
                        source_turn_id=item.turn_id,
                        source_role=draft.source_role,
                        confidence=draft.confidence,
                        reason=draft.reason,
                        time_confidence=draft.time_confidence,
                        caused_by=draft.caused_by,
                        schema_version=1,
                    )
                    event_entities = derive_event_entities(event)
                    self.store.append_event(tx, event)
                    self.store.append_event_entities(tx, event.id, event_entities)
                    dirty_rows.extend(derive_dirty_rows(event, event_entities))
                    dirty_rows.extend(
                        derive_cascade_dirty_rows_for_entity_event(
                            event,
                            self.store.related_owner_ids_for_entity(event.data["id"])
                            if event.type.startswith("entity.")
                            else [],
                        )
                    )

                if superseded_run_ids:
                    dirty_rows.extend(
                        _dirty_rows_for_owners(
                            superseded_owner_ids,
                            processed_at=processed_at,
                            reason=f"supersede:{run_id}",
                        )
                    )

                self.store.mark_dirty(tx, dirty_rows)
        except Exception as exc:
            self._record_failed_run(item, processed_at, str(exc))
            raise
        return True

    def _record_failed_run(self, item: QueueItem, processed_at, error: str) -> None:
        run = ExtractionRun(
            id=str(uuid4()),
            source_turn_id=item.turn_id,
            extractor_version=self.extractor.version,
            observed_at=item.observed_at,
            processed_at=processed_at,
            status="FAILED",
            error=error,
            event_count=0,
            superseded_at=None,
            projection_version=None,
        )
        with self.store.transaction() as tx:
            self.store.append_extraction_run(tx, run)


def _dirty_rows_for_owners(
    owner_ids: list[str],
    *,
    processed_at,
    reason: str,
) -> list[DirtyRangeRow]:
    if not owner_ids:
        return []
    recorded_at = to_rfc3339(processed_at)
    created_at = to_rfc3339(utcnow())
    return [
        (
            str(uuid4()),
            owner_id,
            recorded_at,
            None,
            reason,
            created_at,
        )
        for owner_id in owner_ids
    ]
