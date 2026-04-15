from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias

from engram.semantic import embedding_from_blob
from engram.time_utils import from_rfc3339, to_rfc3339, utcnow
from engram.types import Entity, Event, ExtractionRun

DirtyRangeRow: TypeAlias = tuple[str, str, str, str | None, str, str]


@dataclass(slots=True)
class FoldedEntityState:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    supporting_event_ids: list[str]
    created_recorded_at: datetime | None
    updated_recorded_at: datetime | None
    active: bool


@dataclass(slots=True)
class FoldedValidEntityState:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    unknown_attrs: list[str]
    supporting_event_ids: list[str]
    active: bool


def open_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = FULL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    schema = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()
    return conn


class EventStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def next_seq(self, tx: sqlite3.Connection) -> int:
        row = tx.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events").fetchone()
        return int(row[0])

    def append_event(self, tx: sqlite3.Connection, event: Event) -> None:
        tx.execute(
            """
            INSERT INTO events (
                id, seq, observed_at, effective_at_start, effective_at_end,
                recorded_at, type, data, extraction_run_id, source_turn_id,
                source_role, confidence, reason, time_confidence, caused_by, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.seq,
                to_rfc3339(event.observed_at),
                to_rfc3339(event.effective_at_start) if event.effective_at_start else None,
                to_rfc3339(event.effective_at_end) if event.effective_at_end else None,
                to_rfc3339(event.recorded_at),
                event.type,
                json.dumps(event.data, ensure_ascii=False),
                event.extraction_run_id,
                event.source_turn_id,
                event.source_role,
                event.confidence,
                event.reason,
                event.time_confidence,
                event.caused_by,
                event.schema_version,
            ),
        )

    def append_event_entities(
        self,
        tx: sqlite3.Connection,
        event_id: str,
        entities: list[tuple[str, str]],
    ) -> None:
        if not entities:
            return
        tx.executemany(
            "INSERT INTO event_entities(event_id, entity_id, role) VALUES (?, ?, ?)",
            [(event_id, entity_id, role) for entity_id, role in entities],
        )

    def mark_dirty(
        self,
        tx: sqlite3.Connection,
        rows: list[DirtyRangeRow],
    ) -> None:
        if not rows:
            return
        tx.executemany(
            """
            INSERT INTO dirty_ranges(
                id, owner_id, from_recorded_at, from_effective_at, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def count_events(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return int(row[0])

    def count_extraction_runs(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()
        return int(row[0])

    def count_dirty_ranges(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM dirty_ranges").fetchone()
        return int(row[0])

    def count_vec_events(self, embedder_version: str | None = None) -> int:
        if embedder_version is None:
            row = self.conn.execute("SELECT COUNT(*) FROM vec_events").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM vec_events WHERE embedder_version = ?",
                (embedder_version,),
            ).fetchone()
        return int(row[0])

    def dirty_owner_ids(self) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT owner_id
            FROM dirty_ranges
            ORDER BY owner_id ASC
            """
        ).fetchall()
        return [str(row[0]) for row in rows]

    def clear_dirty_ranges_for_owners(
        self,
        tx: sqlite3.Connection,
        owners: list[str],
    ) -> None:
        if not owners:
            return
        placeholders = ",".join("?" for _ in owners)
        tx.execute(
            f"DELETE FROM dirty_ranges WHERE owner_id IN ({placeholders})",
            owners,
        )

    def clear_all_dirty_ranges(self, tx: sqlite3.Connection) -> None:
        tx.execute("DELETE FROM dirty_ranges")

    def all_entity_ids(self) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT entity_id
            FROM event_entities
            ORDER BY entity_id ASC
            """
        ).fetchall()
        return [str(row[0]) for row in rows]

    def events_missing_embeddings(self, embedder_version: str) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT e.*
            FROM events e
            LEFT JOIN vec_events v
              ON v.event_id = e.id
             AND v.embedder_version = ?
            WHERE v.event_id IS NULL
            ORDER BY e.seq ASC
            """,
            (embedder_version,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def missing_event_embedding_ids(self, embedder_version: str) -> list[str]:
        return [event.id for event in self.events_missing_embeddings(embedder_version)]

    def append_event_embeddings(
        self,
        tx: sqlite3.Connection,
        rows: list[tuple[str, str, int, bytes, str]],
    ) -> None:
        if not rows:
            return
        tx.executemany(
            """
            INSERT OR REPLACE INTO vec_events(
                event_id, embedder_version, dim, embedding, indexed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    def successful_source_turn_ids(self, extractor_version: str) -> set[str]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT source_turn_id
            FROM extraction_runs
            WHERE status = 'SUCCEEDED'
              AND extractor_version = ?
              AND superseded_at IS NULL
            """
            ,
            (extractor_version,),
        ).fetchall()
        return {str(row[0]) for row in rows}

    def has_successful_extraction_run(self, source_turn_id: str, extractor_version: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM extraction_runs
            WHERE source_turn_id = ?
              AND extractor_version = ?
              AND status = 'SUCCEEDED'
              AND superseded_at IS NULL
            LIMIT 1
            """,
            (source_turn_id, extractor_version),
        ).fetchone()
        return row is not None

    def active_successful_runs_for_turn(self, source_turn_id: str) -> list[ExtractionRun]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM extraction_runs
            WHERE source_turn_id = ?
              AND status = 'SUCCEEDED'
              AND superseded_at IS NULL
            ORDER BY processed_at ASC, id ASC
            """,
            (source_turn_id,),
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def supersede_runs(
        self,
        tx: sqlite3.Connection,
        *,
        old_run_ids: list[str],
        new_run_id: str,
        superseded_at: str,
    ) -> None:
        if not old_run_ids:
            return
        placeholders = ",".join("?" for _ in old_run_ids)
        tx.execute(
            f"""
            UPDATE extraction_runs
            SET superseded_at = ?
            WHERE id IN ({placeholders})
              AND superseded_at IS NULL
            """,
            [superseded_at, *old_run_ids],
        )
        tx.executemany(
            """
            INSERT INTO superseded_runs(old_run_id, new_run_id, superseded_at)
            VALUES (?, ?, ?)
            """,
            [(old_run_id, new_run_id, superseded_at) for old_run_id in old_run_ids],
        )

    def list_superseded_runs(self) -> list[dict[str, str]]:
        rows = self.conn.execute(
            """
            SELECT old_run_id, new_run_id, superseded_at
            FROM superseded_runs
            ORDER BY superseded_at ASC, old_run_id ASC
            """
        ).fetchall()
        return [
            {
                "old_run_id": str(row["old_run_id"]),
                "new_run_id": str(row["new_run_id"]),
                "superseded_at": str(row["superseded_at"]),
            }
            for row in rows
        ]

    def entity_owner_ids_for_runs(self, run_ids: list[str]) -> list[str]:
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT ee.entity_id
            FROM event_entities ee
            JOIN events e ON e.id = ee.event_id
            WHERE e.extraction_run_id IN ({placeholders})
            ORDER BY ee.entity_id ASC
            """,
            run_ids,
        ).fetchall()
        return [str(row[0]) for row in rows]

    def append_extraction_run(self, tx: sqlite3.Connection, run: ExtractionRun) -> None:
        tx.execute(
            """
            INSERT INTO extraction_runs(
                id, source_turn_id, extractor_version, observed_at, processed_at,
                status, error, event_count, superseded_at, projection_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.source_turn_id,
                run.extractor_version,
                to_rfc3339(run.observed_at),
                to_rfc3339(run.processed_at),
                run.status,
                run.error,
                run.event_count,
                to_rfc3339(run.superseded_at) if run.superseded_at else None,
                run.projection_version,
            ),
        )

    def list_extraction_runs(self) -> list[ExtractionRun]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM extraction_runs
            ORDER BY processed_at ASC, id ASC
            """
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def entity_events_known_visible_at(self, entity_id: str, recorded_at: str) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT e.*
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            LEFT JOIN extraction_runs r ON r.id = e.extraction_run_id
            WHERE ee.entity_id = ?
              AND e.recorded_at <= ?
              AND (
                e.extraction_run_id IS NULL
                OR (
                    r.status = 'SUCCEEDED'
                    AND r.processed_at <= ?
                    AND (r.superseded_at IS NULL OR r.superseded_at > ?)
                )
              )
            ORDER BY e.recorded_at ASC, e.seq ASC
            """,
            (entity_id, recorded_at, recorded_at, recorded_at),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def entity_events_valid_visible(self, entity_id: str) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT e.*
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            LEFT JOIN extraction_runs r ON r.id = e.extraction_run_id
            WHERE ee.entity_id = ?
              AND (
                e.extraction_run_id IS NULL
                OR (
                    r.status = 'SUCCEEDED'
                    AND r.superseded_at IS NULL
                )
              )
            ORDER BY e.recorded_at ASC, e.seq ASC
            """,
            (entity_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def entity_events_known_current(self, entity_id: str) -> list[Event]:
        return self.entity_events_valid_visible(entity_id)

    def entity_events(self, entity_id: str) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT e.*
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            WHERE ee.entity_id = ?
            ORDER BY e.recorded_at ASC, e.seq ASC
            """,
            (entity_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def visible_events_known(self, recorded_at: str, from_recorded_at: str | None = None) -> list[Event]:
        if from_recorded_at is None:
            rows = self.conn.execute(
                """
                SELECT events.*
                FROM events
                LEFT JOIN extraction_runs r ON r.id = events.extraction_run_id
                WHERE events.recorded_at <= ?
                  AND (
                    events.extraction_run_id IS NULL
                    OR (
                        r.status = 'SUCCEEDED'
                        AND r.processed_at <= ?
                        AND (r.superseded_at IS NULL OR r.superseded_at > ?)
                    )
                  )
                ORDER BY events.recorded_at ASC, events.seq ASC
                """,
                (recorded_at, recorded_at, recorded_at),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT events.*
                FROM events
                LEFT JOIN extraction_runs r ON r.id = events.extraction_run_id
                WHERE events.recorded_at >= ?
                  AND events.recorded_at <= ?
                  AND (
                    events.extraction_run_id IS NULL
                    OR (
                        r.status = 'SUCCEEDED'
                        AND r.processed_at <= ?
                        AND (r.superseded_at IS NULL OR r.superseded_at > ?)
                    )
                  )
                ORDER BY events.recorded_at ASC, events.seq ASC
                """,
                (from_recorded_at, recorded_at, recorded_at, recorded_at),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def visible_events_valid(self) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT events.*
            FROM events
            LEFT JOIN extraction_runs r ON r.id = events.extraction_run_id
            WHERE (
                events.extraction_run_id IS NULL
                OR (
                    r.status = 'SUCCEEDED'
                    AND r.superseded_at IS NULL
                )
            )
            ORDER BY events.recorded_at ASC, events.seq ASC
            """
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def event_entity_ids_for_events(self, event_ids: list[str]) -> dict[str, list[str]]:
        if not event_ids:
            return {}
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.conn.execute(
            f"""
            SELECT event_id, entity_id
            FROM event_entities
            WHERE event_id IN ({placeholders})
            ORDER BY event_id ASC, entity_id ASC
            """,
            event_ids,
        ).fetchall()
        mapping: dict[str, list[str]] = {}
        for row in rows:
            mapping.setdefault(str(row["event_id"]), []).append(str(row["entity_id"]))
        return mapping

    def events_by_ids(self, event_ids: list[str]) -> list[Event]:
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.conn.execute(
            f"""
            SELECT *
            FROM events
            WHERE id IN ({placeholders})
            ORDER BY recorded_at ASC, seq ASC
            """,
            event_ids,
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def event_embeddings_for_ids(
        self,
        event_ids: list[str],
        *,
        embedder_version: str,
    ) -> dict[str, list[float]]:
        if not event_ids:
            return {}
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.conn.execute(
            f"""
            SELECT event_id, dim, embedding
            FROM vec_events
            WHERE embedder_version = ?
              AND event_id IN ({placeholders})
            ORDER BY event_id ASC
            """,
            [embedder_version, *event_ids],
        ).fetchall()
        return {
            str(row["event_id"]): embedding_from_blob(row["embedding"], dim=int(row["dim"]))
            for row in rows
        }

    def materialize_current_entity(self, entity_id: str) -> Entity | None:
        folded = self.fold_entity_events(entity_id, self.entity_events_known_current(entity_id))
        if folded is None:
            return None
        return Entity(
            id=folded.entity_id,
            type=folded.entity_type,
            attrs=dict(folded.attrs),
            created_recorded_at=folded.created_recorded_at,
            updated_recorded_at=folded.updated_recorded_at,
        )

    def fold_entity_events(self, entity_id: str, events: list[Event]) -> FoldedEntityState | None:
        entity_type = "unknown"
        attrs: dict[str, Any] = {}
        supporting_event_ids: list[str] = []
        created_at = None
        updated_at = None
        active = False

        for event in events:
            if not event.type.startswith("entity.") or event.data["id"] != entity_id:
                continue
            supporting_event_ids.append(event.id)
            if event.type == "entity.create":
                entity_type = event.data["type"]
                attrs = dict(event.data["attrs"])
                created_at = event.recorded_at
                updated_at = event.recorded_at
                active = True
            elif event.type == "entity.update":
                if not active:
                    created_at = event.recorded_at
                    active = True
                attrs.update(event.data["attrs"])
                updated_at = event.recorded_at
            elif event.type == "entity.delete":
                attrs = {}
                created_at = None
                updated_at = None
                active = False

        if not active or created_at is None or updated_at is None:
            return None

        return FoldedEntityState(
            entity_id=entity_id,
            entity_type=entity_type,
            attrs=dict(attrs),
            supporting_event_ids=supporting_event_ids,
            created_recorded_at=created_at,
            updated_recorded_at=updated_at,
            active=active,
        )

    def fold_entity_events_valid_at(
        self,
        entity_id: str,
        at: datetime,
        *,
        events: list[Event] | None = None,
    ) -> FoldedValidEntityState | None:
        events = sorted(events if events is not None else self.entity_events_valid_visible(entity_id), key=valid_event_sort_key)
        entity_type = "unknown"
        attrs: dict[str, Any] = {}
        unknown_attrs: list[str] = []
        supporting_event_ids: list[str] = []
        active = False

        for event in events:
            if not event.type.startswith("entity.") or event.data["id"] != entity_id:
                continue
            if event.type == "entity.create":
                entity_type = event.data["type"]

            if not covers_valid_time(event, at):
                if _has_unknown_effective_time(event):
                    unknown_attrs = _merge_unknown_attrs(unknown_attrs, event.data.get("attrs", {}).keys())
                continue

            supporting_event_ids.append(event.id)
            if event.type == "entity.create":
                attrs = dict(event.data["attrs"])
                active = True
            elif event.type == "entity.update":
                if not active:
                    active = True
                attrs.update(event.data["attrs"])
            elif event.type == "entity.delete":
                attrs = {}
                active = False
                unknown_attrs = []

        if not active and not unknown_attrs:
            return None

        return FoldedValidEntityState(
            entity_id=entity_id,
            entity_type=entity_type,
            attrs=dict(attrs),
            unknown_attrs=unknown_attrs,
            supporting_event_ids=supporting_event_ids,
            active=active,
        )

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            seq=int(row["seq"]),
            observed_at=from_rfc3339(row["observed_at"]),
            effective_at_start=from_rfc3339(row["effective_at_start"]) if row["effective_at_start"] else None,
            effective_at_end=from_rfc3339(row["effective_at_end"]) if row["effective_at_end"] else None,
            recorded_at=from_rfc3339(row["recorded_at"]),
            type=row["type"],
            data=json.loads(row["data"]),
            extraction_run_id=row["extraction_run_id"],
            source_turn_id=row["source_turn_id"],
            source_role=row["source_role"],
            confidence=row["confidence"],
            reason=row["reason"],
            time_confidence=row["time_confidence"],
            caused_by=row["caused_by"],
            schema_version=int(row["schema_version"]),
        )

    def _row_to_run(self, row: sqlite3.Row) -> ExtractionRun:
        return ExtractionRun(
            id=row["id"],
            source_turn_id=row["source_turn_id"],
            extractor_version=row["extractor_version"],
            observed_at=from_rfc3339(row["observed_at"]),
            processed_at=from_rfc3339(row["processed_at"]),
            status=row["status"],
            error=row["error"],
            event_count=int(row["event_count"]),
            superseded_at=from_rfc3339(row["superseded_at"]) if row["superseded_at"] else None,
            projection_version=int(row["projection_version"]) if row["projection_version"] is not None else None,
        )


def covers_valid_time(event: Event, at: datetime) -> bool:
    if _has_unknown_effective_time(event):
        return False
    start = event.effective_at_start
    end = event.effective_at_end
    if start is None:
        return False
    if start > at:
        return False
    if end is not None and at >= end:
        return False
    return True


def overlaps_valid_time_window(
    event: Event,
    start_at: datetime,
    end_at: datetime,
) -> bool:
    if _has_unknown_effective_time(event):
        return False

    start = event.effective_at_start
    end = event.effective_at_end
    if start is None:
        return False
    if end is not None and end <= start_at:
        return False
    if start >= end_at:
        return False
    return True


def _has_unknown_effective_time(event: Event) -> bool:
    return event.effective_at_start is None


def _merge_unknown_attrs(existing: list[str], new_keys) -> list[str]:
    seen = set(existing)
    merged = list(existing)
    for key in new_keys:
        if key in seen:
            continue
        seen.add(key)
        merged.append(str(key))
    return merged


def valid_event_sort_key(event: Event) -> tuple[bool, datetime, datetime, int]:
    return (
        event.effective_at_start is None,
        event.effective_at_start or datetime.max.replace(tzinfo=UTC),
        event.recorded_at,
        event.seq,
    )
