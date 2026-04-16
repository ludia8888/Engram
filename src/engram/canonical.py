from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .data_quality import DataQualityManager
from .errors import ValidationError
from .retry import RetryPolicy, RetryState
from .event_ops import (
    derive_cascade_dirty_rows_for_entity_event,
    derive_dirty_rows,
    derive_event_entities,
    validate_event,
)
from .search_terms import event_search_terms
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
    def __init__(self, store: EventStore, extractor: Extractor, data_quality: DataQualityManager):
        self.store = store
        self.extractor = extractor
        self.data_quality = data_quality

    def process(self, item: QueueItem, *, force: bool = False) -> bool:
        if not force and self.store.has_successful_extraction_run(item.turn_id, self.extractor.version):
            return False

        processed_at = utcnow()
        try:
            drafts = self.extractor.extract(item)
            quality_batch = self.data_quality.process_drafts(
                drafts,
                observed_at=item.observed_at,
                source_turn_id=item.turn_id,
            )
            drafts = quality_batch.drafts
        except Exception as exc:
            self._record_failed_run(item, processed_at, str(exc))
            raise

        try:
            for draft in drafts:
                validate_event(draft.type, draft.data)
                if (
                    draft.caused_by is not None
                    and not draft.caused_by.startswith(_BATCH_REF_PREFIX)
                    and not self.store.event_exists(draft.caused_by)
                ):
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
                batch_event_ids: list[str] = []
                for offset, draft in enumerate(drafts):
                    caused_by = _resolve_batch_caused_by(draft.caused_by, batch_event_ids)
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
                        caused_by=caused_by,
                        schema_version=1,
                    )
                    batch_event_ids.append(event.id)
                    event_entities = derive_event_entities(event)
                    self.store.append_event(tx, event)
                    self.store.append_event_entities(tx, event.id, event_entities)
                    self.store.append_event_search_terms(tx, event.id, event_search_terms(event))
                    dirty_rows.extend(derive_dirty_rows(event, event_entities))
                    dirty_rows.extend(
                        derive_cascade_dirty_rows_for_entity_event(
                            event,
                            self.store.related_owner_ids_for_entity(event.data["id"])
                            if event.type.startswith("entity.")
                            else [],
                        )
                    )
                if quality_batch.aliases:
                    self.store.append_entity_alias_rows(
                        tx,
                        [
                            (
                                entity_id,
                                entity_type,
                                alias,
                                normalized_alias,
                                alias_kind,
                                processed_at,
                            )
                            for entity_id, entity_type, alias, normalized_alias, alias_kind in quality_batch.aliases
                        ],
                    )
                if quality_batch.duplicate_candidates:
                    self.store.append_duplicate_candidates(tx, quality_batch.duplicate_candidates)

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

    def process_with_retry(
        self,
        item: QueueItem,
        policy: RetryPolicy,
    ) -> tuple[bool, RetryState | None]:
        logger = logging.getLogger(__name__)
        prior_failures = self.store.count_failed_runs_for_turn(
            item.turn_id, self.extractor.version,
        )
        if prior_failures >= policy.max_retries:
            logger.error("permanently failed turn %s after %d prior attempts", item.turn_id, prior_failures)
            return False, None
        try:
            success = self.process(item)
            return success, None
        except Exception as exc:
            attempt = prior_failures + 1
            if attempt >= policy.max_retries:
                logger.error("permanently failed turn %s after %d attempts: %s", item.turn_id, attempt, exc)
                return False, None
            state = RetryState(turn_id=item.turn_id, attempt=attempt, last_error=str(exc))
            state.schedule_next(policy)
            logger.warning("retrying turn %s (attempt %d/%d): %s", item.turn_id, attempt, policy.max_retries, exc)
            return False, state

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


_BATCH_REF_PREFIX = "__batch_ref:"


def _resolve_batch_caused_by(
    raw_caused_by: str | None,
    batch_event_ids: list[str],
) -> str | None:
    if raw_caused_by is None:
        return None
    if not raw_caused_by.startswith(_BATCH_REF_PREFIX):
        return raw_caused_by
    try:
        index = int(raw_caused_by[len(_BATCH_REF_PREFIX):])
    except ValueError:
        return None
    if 0 <= index < len(batch_event_ids):
        return batch_event_ids[index]
    return None


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
