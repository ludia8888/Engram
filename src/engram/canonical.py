from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .event_ops import derive_dirty_rows, derive_event_entities, validate_event
from .storage.store import DirtyRangeRow, EventStore
from .time_utils import utcnow
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

    def process(self, item: QueueItem) -> bool:
        if self.store.has_successful_extraction_run(item.turn_id, self.extractor.version):
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
            run_id = str(uuid4())
            with self.store.transaction() as tx:
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
                        caused_by=None,
                        schema_version=1,
                    )
                    event_entities = derive_event_entities(event)
                    self.store.append_event(tx, event)
                    self.store.append_event_entities(tx, event.id, event_entities)
                    dirty_rows.extend(derive_dirty_rows(event, event_entities))

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
