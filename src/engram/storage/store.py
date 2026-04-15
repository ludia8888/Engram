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
from engram.types import Entity, Event, ExtractionRun, RelationEdge

from .entity_fold import (
    FoldedEntityState,
    FoldedValidEntityState,
    fold_entity_events,
    fold_entity_events_valid_at,
)
from .relation_fold import (
    entity_active_intervals as _entity_active_intervals,
    fold_relation_edges as _fold_relation_edges,
    fold_relation_edges_in_window as _fold_relation_edges_in_window,
    intervals_overlap as _intervals_overlap,
    min_optional as _min_optional,
    relation_window_states as _relation_window_states,
)
from .temporal import (
    _has_unknown_effective_time,
    covers_valid_time,
    overlaps_valid_time_window,
    valid_event_sort_key,
)

DirtyRangeRow: TypeAlias = tuple[str, str, str, str | None, str, str]


@dataclass(slots=True)
class RelationWindowQueryCache:
    interval_cache: dict[str, list[tuple[datetime, datetime | None]]]
    relation_key_cache: dict[str, set[tuple[str, str, str]]]
    source_relation_events: dict[str, list[Event]]


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

    def count_event_search_terms(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM event_search_terms").fetchone()
        return int(row[0])

    def event_exists(self, event_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM events WHERE id = ? LIMIT 1",
            (event_id,),
        ).fetchone()
        return row is not None

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

    def events_missing_search_terms(self) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT e.*
            FROM events e
            LEFT JOIN event_search_terms t ON t.event_id = e.id
            WHERE t.event_id IS NULL
            ORDER BY e.seq ASC
            """
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def append_event_search_terms(
        self,
        tx: sqlite3.Connection,
        event_id: str,
        terms: list[str],
    ) -> None:
        if not terms:
            return
        tx.executemany(
            """
            INSERT OR REPLACE INTO event_search_terms(event_id, term)
            VALUES (?, ?)
            """,
            [(event_id, term) for term in terms],
        )

    def candidate_event_ids_for_search_terms(self, terms: list[str]) -> list[str]:
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT event_id
            FROM event_search_terms
            WHERE term IN ({placeholders})
            ORDER BY event_id ASC
            """,
            terms,
        ).fetchall()
        return [str(row["event_id"]) for row in rows]

    def event_ids_with_embeddings(self, embedder_version: str) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT event_id
            FROM vec_events
            WHERE embedder_version = ?
            ORDER BY event_id ASC
            """,
            (embedder_version,),
        ).fetchall()
        return [str(row["event_id"]) for row in rows]

    def event_ids_by_caused_by(self, caused_by_ids: list[str]) -> list[str]:
        if not caused_by_ids:
            return []
        placeholders = ",".join("?" for _ in caused_by_ids)
        rows = self.conn.execute(
            f"""
            SELECT id
            FROM events
            WHERE caused_by IN ({placeholders})
            ORDER BY id ASC
            """,
            caused_by_ids,
        ).fetchall()
        return [str(row["id"]) for row in rows]

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

    def related_owner_ids_for_entity(self, entity_id: str) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT CASE
                WHEN json_extract(e.data, '$.source') = ? THEN json_extract(e.data, '$.target')
                WHEN json_extract(e.data, '$.target') = ? THEN json_extract(e.data, '$.source')
            END AS other_entity_id
            FROM events e
            LEFT JOIN extraction_runs r ON r.id = e.extraction_run_id
            WHERE e.type LIKE 'relation.%'
              AND (
                    json_extract(e.data, '$.source') = ?
                 OR json_extract(e.data, '$.target') = ?
              )
              AND (
                e.extraction_run_id IS NULL
                OR (
                    r.status = 'SUCCEEDED'
                    AND r.superseded_at IS NULL
                )
              )
            ORDER BY other_entity_id ASC
            """,
            (entity_id, entity_id, entity_id, entity_id),
        ).fetchall()
        return [str(row["other_entity_id"]) for row in rows if row["other_entity_id"] is not None]

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

    def entity_events_known_visible_for_entities(
        self,
        entity_ids: list[str],
        recorded_at: str,
    ) -> dict[str, list[Event]]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.conn.execute(
            f"""
            SELECT ee.entity_id AS lookup_entity_id, e.*
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            LEFT JOIN extraction_runs r ON r.id = e.extraction_run_id
            WHERE ee.entity_id IN ({placeholders})
              AND e.recorded_at <= ?
              AND (
                e.extraction_run_id IS NULL
                OR (
                    r.status = 'SUCCEEDED'
                    AND r.processed_at <= ?
                    AND (r.superseded_at IS NULL OR r.superseded_at > ?)
                )
              )
            ORDER BY ee.entity_id ASC, e.recorded_at ASC, e.seq ASC
            """,
            [*entity_ids, recorded_at, recorded_at, recorded_at],
        ).fetchall()
        grouped: dict[str, list[Event]] = {entity_id: [] for entity_id in entity_ids}
        for row in rows:
            grouped.setdefault(str(row["lookup_entity_id"]), []).append(self._row_to_event(row))
        return grouped

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

    def entity_events_valid_visible_for_entities(self, entity_ids: list[str]) -> dict[str, list[Event]]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.conn.execute(
            f"""
            SELECT ee.entity_id AS lookup_entity_id, e.*
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            LEFT JOIN extraction_runs r ON r.id = e.extraction_run_id
            WHERE ee.entity_id IN ({placeholders})
              AND (
                e.extraction_run_id IS NULL
                OR (
                    r.status = 'SUCCEEDED'
                    AND r.superseded_at IS NULL
                )
              )
            ORDER BY ee.entity_id ASC, e.recorded_at ASC, e.seq ASC
            """,
            entity_ids,
        ).fetchall()
        grouped: dict[str, list[Event]] = {entity_id: [] for entity_id in entity_ids}
        for row in rows:
            grouped.setdefault(str(row["lookup_entity_id"]), []).append(self._row_to_event(row))
        return grouped

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

    def visible_events_known(
        self,
        recorded_at: str,
        from_recorded_at: str | None = None,
        event_ids: list[str] | None = None,
    ) -> list[Event]:
        event_filter = ""
        params: list[str] = []
        if event_ids:
            placeholders = ",".join("?" for _ in event_ids)
            event_filter = f" AND events.id IN ({placeholders})"
            params.extend(event_ids)
        if from_recorded_at is None:
            rows = self.conn.execute(
                f"""
                SELECT events.*
                FROM events
                LEFT JOIN extraction_runs r ON r.id = events.extraction_run_id
                WHERE events.recorded_at <= ?
                  {event_filter}
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
                (recorded_at, *params, recorded_at, recorded_at),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"""
                SELECT events.*
                FROM events
                LEFT JOIN extraction_runs r ON r.id = events.extraction_run_id
                WHERE events.recorded_at >= ?
                  AND events.recorded_at <= ?
                  {event_filter}
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
                (from_recorded_at, recorded_at, *params, recorded_at, recorded_at),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def visible_events_valid(self, event_ids: list[str] | None = None) -> list[Event]:
        event_filter = ""
        params: list[str] = []
        if event_ids:
            placeholders = ",".join("?" for _ in event_ids)
            event_filter = f" AND events.id IN ({placeholders})"
            params.extend(event_ids)
        rows = self.conn.execute(
            f"""
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
            {event_filter}
            ORDER BY events.recorded_at ASC, events.seq ASC
            """,
            params,
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

    def event_by_id(self, event_id: str) -> Event | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM events
            WHERE id = ?
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

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

    def event_ids_with_embeddings_for_entities(
        self,
        entity_ids: list[str],
        *,
        embedder_version: str,
    ) -> list[str]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT ee.event_id
            FROM event_entities ee
            JOIN vec_events v ON v.event_id = ee.event_id
            WHERE ee.entity_id IN ({placeholders})
              AND v.embedder_version = ?
            ORDER BY ee.event_id ASC
            """,
            [*entity_ids, embedder_version],
        ).fetchall()
        return [str(row["event_id"]) for row in rows]

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

    def materialize_current_relations(self, entity_id: str) -> list[RelationEdge]:
        return self.relation_edges_known_at(entity_id, to_rfc3339(utcnow()))

    def fold_entities_known_at(
        self,
        entity_ids: list[str],
        recorded_at: str,
    ) -> dict[str, FoldedEntityState | None]:
        events_by_entity = self.entity_events_known_visible_for_entities(entity_ids, recorded_at)
        return {
            entity_id: self.fold_entity_events(entity_id, events_by_entity.get(entity_id, []))
            for entity_id in entity_ids
        }

    def fold_entities_valid_at(
        self,
        entity_ids: list[str],
        at: datetime,
    ) -> dict[str, FoldedValidEntityState | None]:
        events_by_entity = self.entity_events_valid_visible_for_entities(entity_ids)
        return {
            entity_id: self.fold_entity_events_valid_at(
                entity_id,
                at,
                events=events_by_entity.get(entity_id, []),
            )
            for entity_id in entity_ids
        }

    def relation_edges_known_at(self, entity_id: str, recorded_at: str) -> list[RelationEdge]:
        active_cache: dict[str, bool] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_cache:
                active_cache[candidate_id] = self._entity_is_known_active_at(candidate_id, recorded_at)
            return active_cache[candidate_id]

        return self.fold_relation_edges(
            entity_id,
            self.relation_events_known_visible_at(entity_id, recorded_at),
            endpoint_active=endpoint_active,
        )

    def relation_edges_known_at_many(self, entity_ids: list[str], recorded_at: str) -> dict[str, list[RelationEdge]]:
        events_by_entity = self.entity_events_known_visible_for_entities(entity_ids, recorded_at)
        active_cache: dict[str, bool] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_cache:
                active_cache[candidate_id] = self._entity_is_known_active_at(candidate_id, recorded_at)
            return active_cache[candidate_id]

        return {
            entity_id: self.fold_relation_edges(
                entity_id,
                [event for event in events_by_entity.get(entity_id, []) if event.type.startswith("relation.")],
                endpoint_active=endpoint_active,
            )
            for entity_id in entity_ids
        }

    def relation_edges_valid_at(self, entity_id: str, at: datetime) -> list[RelationEdge]:
        active_cache: dict[str, bool] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_cache:
                active_cache[candidate_id] = self._entity_is_valid_active_at(candidate_id, at)
            return active_cache[candidate_id]

        return self.fold_relation_edges_valid_at(
            entity_id,
            at,
            events=self.relation_events_valid_visible(entity_id),
            endpoint_active=endpoint_active,
        )

    def relation_edges_valid_at_many(self, entity_ids: list[str], at: datetime) -> dict[str, list[RelationEdge]]:
        events_by_entity = self.entity_events_valid_visible_for_entities(entity_ids)
        active_cache: dict[str, bool] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_cache:
                active_cache[candidate_id] = self._entity_is_valid_active_at(candidate_id, at)
            return active_cache[candidate_id]

        return {
            entity_id: self.fold_relation_edges_valid_at(
                entity_id,
                at,
                events=[event for event in events_by_entity.get(entity_id, []) if event.type.startswith("relation.")],
                endpoint_active=endpoint_active,
            )
            for entity_id in entity_ids
        }

    def relation_edges_valid_in_window(
        self,
        entity_id: str,
        start_at: datetime,
        end_at: datetime,
    ) -> list[RelationEdge]:
        return self.relation_edges_valid_in_window_many([entity_id], start_at, end_at).get(entity_id, [])

    def relation_edges_valid_in_window_many(
        self,
        entity_ids: list[str],
        start_at: datetime,
        end_at: datetime,
        *,
        query_cache: RelationWindowQueryCache | None = None,
    ) -> dict[str, list[RelationEdge]]:
        if not entity_ids:
            return {}
        cache = query_cache or self.build_relation_window_query_cache(start_at, end_at, source_entity_ids=entity_ids)
        self._ensure_relation_window_sources(cache, entity_ids)

        def endpoint_active_in_window(source: str, target: str, overlap_start: datetime, overlap_end: datetime) -> bool:
            return self._entities_are_valid_together_in_window(
                source,
                target,
                overlap_start,
                overlap_end,
                interval_cache=cache.interval_cache,
            )

        return {
            entity_id: self.fold_relation_edges_valid_in_window(
                entity_id,
                start_at,
                end_at,
                events=cache.source_relation_events.get(entity_id, []),
                endpoint_active_in_window=endpoint_active_in_window,
            )
            for entity_id in entity_ids
        }

    def relation_events_known_visible_at(self, entity_id: str, recorded_at: str) -> list[Event]:
        return [
            event
            for event in self.entity_events_known_visible_at(entity_id, recorded_at)
            if event.type.startswith("relation.")
        ]

    def relation_events_valid_visible(self, entity_id: str) -> list[Event]:
        return [
            event
            for event in self.entity_events_valid_visible(entity_id)
            if event.type.startswith("relation.")
        ]

    def relation_events_valid_visible_for_entities(self, entity_ids: list[str]) -> dict[str, list[Event]]:
        return {
            entity_id: [event for event in events if event.type.startswith("relation.")]
            for entity_id, events in self.entity_events_valid_visible_for_entities(entity_ids).items()
        }

    def fold_entity_events(self, entity_id: str, events: list[Event]) -> FoldedEntityState | None:
        return fold_entity_events(entity_id, events)

    def fold_entity_events_valid_at(
        self,
        entity_id: str,
        at: datetime,
        *,
        events: list[Event] | None = None,
    ) -> FoldedValidEntityState | None:
        return fold_entity_events_valid_at(
            entity_id,
            at,
            events if events is not None else self.entity_events_valid_visible(entity_id),
        )

    def fold_relation_edges(
        self,
        entity_id: str,
        events: list[Event],
        *,
        endpoint_active=None,
    ) -> list[RelationEdge]:
        return _fold_relation_edges(entity_id, events, endpoint_active=endpoint_active)

    def fold_relation_edges_valid_at(
        self,
        entity_id: str,
        at: datetime,
        *,
        events: list[Event] | None = None,
        endpoint_active=None,
    ) -> list[RelationEdge]:
        visible_events = [
            event
            for event in (events if events is not None else self.entity_events_valid_visible(entity_id))
            if covers_valid_time(event, at)
        ]
        return _fold_relation_edges(entity_id, visible_events, endpoint_active=endpoint_active)

    def fold_relation_edges_valid_in_window(
        self,
        entity_id: str,
        start_at: datetime,
        end_at: datetime,
        *,
        events: list[Event] | None = None,
        endpoint_active_in_window=None,
    ) -> list[RelationEdge]:
        return _fold_relation_edges_in_window(
            entity_id,
            events if events is not None else self.entity_events_valid_visible(entity_id),
            start_at,
            end_at,
            endpoint_active_in_window=endpoint_active_in_window,
        )

    def relation_event_is_live_known(self, event: Event, recorded_at: str) -> bool:
        if not event.type.startswith("relation."):
            return True
        return self._entity_is_known_active_at(str(event.data["source"]), recorded_at) and self._entity_is_known_active_at(
            str(event.data["target"]),
            recorded_at,
        )

    def relation_event_is_live_valid(self, event: Event, at: datetime) -> bool:
        if not event.type.startswith("relation."):
            return True
        return self._relation_key_is_live_valid_at(
            str(event.data["source"]),
            str(event.data["target"]),
            str(event.data["type"]),
            at,
        )

    def relation_event_is_live_valid_in_window(
        self,
        event: Event,
        start_at: datetime,
        end_at: datetime,
    ) -> bool:
        if not event.type.startswith("relation."):
            return True
        return self._relation_key_is_live_valid_in_window(
            str(event.data["source"]),
            str(event.data["target"]),
            str(event.data["type"]),
            start_at,
            end_at,
        )

    def filter_relation_events_live_known(self, events: list[Event], recorded_at: str) -> list[Event]:
        active_cache: dict[str, bool] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_cache:
                active_cache[candidate_id] = self._entity_is_known_active_at(candidate_id, recorded_at)
            return active_cache[candidate_id]

        return [
            event
            for event in events
            if not event.type.startswith("relation.")
            or (
                endpoint_active(str(event.data["source"]))
                and endpoint_active(str(event.data["target"]))
            )
        ]

    def filter_relation_events_live_valid_at(self, events: list[Event], at: datetime) -> list[Event]:
        active_entity_cache: dict[str, bool] = {}
        active_relation_key_cache: dict[str, set[tuple[str, str, str]]] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_entity_cache:
                active_entity_cache[candidate_id] = self._entity_is_valid_active_at(candidate_id, at)
            return active_entity_cache[candidate_id]

        def active_relation_keys_for_source(source: str) -> set[tuple[str, str, str]]:
            if source not in active_relation_key_cache:
                edges = self.fold_relation_edges_valid_at(
                    source,
                    at,
                    events=self.entity_events_valid_visible(source),
                    endpoint_active=endpoint_active,
                )
                active_relation_key_cache[source] = {
                    (source, edge.other_entity_id, edge.relation_type)
                    for edge in edges
                    if edge.direction == "outgoing"
                }
            return active_relation_key_cache[source]

        filtered: list[Event] = []
        for event in events:
            if not event.type.startswith("relation."):
                filtered.append(event)
                continue
            key = (str(event.data["source"]), str(event.data["target"]), str(event.data["type"]))
            if key in active_relation_keys_for_source(key[0]):
                filtered.append(event)
        return filtered

    def filter_relation_events_live_valid_in_window(
        self,
        events: list[Event],
        start_at: datetime,
        end_at: datetime,
        *,
        query_cache: RelationWindowQueryCache | None = None,
    ) -> list[Event]:
        cache = query_cache or self.build_relation_window_query_cache(start_at, end_at)
        source_ids = sorted(
            {
                str(event.data["source"])
                for event in events
                if event.type.startswith("relation.")
            }
        )
        self._ensure_relation_window_sources(cache, source_ids)

        def endpoint_active_in_window(
            overlap_source: str,
            overlap_target: str,
            overlap_start: datetime,
            overlap_end: datetime,
        ) -> bool:
            return self._entities_are_valid_together_in_window(
                overlap_source,
                overlap_target,
                overlap_start,
                overlap_end,
                interval_cache=cache.interval_cache,
            )

        def active_relation_keys_for_source(source: str) -> set[tuple[str, str, str]]:
            if source not in cache.relation_key_cache:
                cache.relation_key_cache[source] = set(
                    _relation_window_states(
                        cache.source_relation_events.get(source, []),
                        start_at,
                        end_at,
                        endpoint_active_in_window=endpoint_active_in_window,
                    ).keys()
                )
            return cache.relation_key_cache[source]

        filtered: list[Event] = []
        for event in events:
            if not event.type.startswith("relation."):
                filtered.append(event)
                continue
            key = (str(event.data["source"]), str(event.data["target"]), str(event.data["type"]))
            if key in active_relation_keys_for_source(key[0]):
                filtered.append(event)
        return filtered

    def build_relation_window_query_cache(
        self,
        start_at: datetime,
        end_at: datetime,
        *,
        source_entity_ids: list[str] | None = None,
    ) -> RelationWindowQueryCache:
        cache = RelationWindowQueryCache(
            interval_cache={},
            relation_key_cache={},
            source_relation_events={},
        )
        if source_entity_ids:
            self._ensure_relation_window_sources(cache, source_entity_ids)
        return cache

    def _ensure_relation_window_sources(
        self,
        cache: RelationWindowQueryCache,
        source_entity_ids: list[str],
    ) -> None:
        missing_ids = [entity_id for entity_id in source_entity_ids if entity_id not in cache.source_relation_events]
        if not missing_ids:
            return
        cache.source_relation_events.update(self.relation_events_valid_visible_for_entities(missing_ids))

    def _entity_is_known_active_at(self, entity_id: str, recorded_at: str) -> bool:
        return self.fold_entity_events(
            entity_id,
            self.entity_events_known_visible_at(entity_id, recorded_at),
        ) is not None

    def _entity_is_valid_active_at(self, entity_id: str, at: datetime) -> bool:
        folded = self.fold_entity_events_valid_at(
            entity_id,
            at,
            events=self.entity_events_valid_visible(entity_id),
        )
        return folded is not None and folded.active

    def _entity_is_valid_in_window(
        self,
        entity_id: str,
        start_at: datetime,
        end_at: datetime,
        *,
        interval_cache: dict[str, list[tuple[datetime, datetime | None]]] | None = None,
    ) -> bool:
        intervals = self._entity_valid_intervals(entity_id, interval_cache=interval_cache)
        return any(_intervals_overlap(interval_start, interval_end, start_at, end_at) for interval_start, interval_end in intervals)

    def _entities_are_valid_together_in_window(
        self,
        source: str,
        target: str,
        start_at: datetime,
        end_at: datetime,
        *,
        interval_cache: dict[str, list[tuple[datetime, datetime | None]]] | None = None,
    ) -> bool:
        source_intervals = self._entity_valid_intervals(source, interval_cache=interval_cache)
        target_intervals = self._entity_valid_intervals(target, interval_cache=interval_cache)
        for source_start, source_end in source_intervals:
            for target_start, target_end in target_intervals:
                overlap_start = max(start_at, source_start, target_start)
                overlap_end = _min_optional(end_at, source_end, target_end)
                if overlap_start < overlap_end:
                    return True
        return False

    def _entity_valid_intervals(
        self,
        entity_id: str,
        *,
        interval_cache: dict[str, list[tuple[datetime, datetime | None]]] | None = None,
    ) -> list[tuple[datetime, datetime | None]]:
        if interval_cache is not None and entity_id in interval_cache:
            return interval_cache[entity_id]
        intervals = _entity_active_intervals(
            entity_id,
            self.entity_events_valid_visible(entity_id),
        )
        if interval_cache is not None:
            interval_cache[entity_id] = intervals
        return intervals

    def _relation_key_is_live_valid_in_window(
        self,
        source: str,
        target: str,
        relation_type: str,
        start_at: datetime,
        end_at: datetime,
    ) -> bool:
        interval_cache: dict[str, list[tuple[datetime, datetime | None]]] = {}

        def endpoint_active_in_window(
            overlap_source: str,
            overlap_target: str,
            overlap_start: datetime,
            overlap_end: datetime,
        ) -> bool:
            return self._entities_are_valid_together_in_window(
                overlap_source,
                overlap_target,
                overlap_start,
                overlap_end,
                interval_cache=interval_cache,
            )

        states = _relation_window_states(
            self.entity_events_valid_visible(source),
            start_at,
            end_at,
            endpoint_active_in_window=endpoint_active_in_window,
        )
        return (source, target, relation_type) in states

    def _relation_key_is_live_valid_at(
        self,
        source: str,
        target: str,
        relation_type: str,
        at: datetime,
    ) -> bool:
        active_cache: dict[str, bool] = {}

        def endpoint_active(candidate_id: str) -> bool:
            if candidate_id not in active_cache:
                active_cache[candidate_id] = self._entity_is_valid_active_at(candidate_id, at)
            return active_cache[candidate_id]

        for edge in self.fold_relation_edges_valid_at(
            source,
            at,
            events=self.entity_events_valid_visible(source),
            endpoint_active=endpoint_active,
        ):
            if edge.direction == "outgoing" and edge.other_entity_id == target and edge.relation_type == relation_type:
                return True
        return False

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


