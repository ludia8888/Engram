from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from engram.time_utils import from_rfc3339, to_rfc3339
from engram.types import Event


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
        rows: list[tuple[str, str | None, str | None, str, str]],
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

    def count_dirty_ranges(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM dirty_ranges").fetchone()
        return int(row[0])

    def entity_events_visible_at(self, entity_id: str, recorded_at: str) -> list[Event]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT e.*
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            WHERE ee.entity_id = ?
              AND e.recorded_at <= ?
            ORDER BY e.recorded_at ASC, e.seq ASC
            """,
            (entity_id, recorded_at),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

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

