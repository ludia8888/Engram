CREATE TABLE IF NOT EXISTS extraction_runs (
    id                 TEXT PRIMARY KEY,
    source_turn_id     TEXT NOT NULL,
    extractor_version  TEXT NOT NULL,
    observed_at        TEXT NOT NULL,
    processed_at       TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('SUCCEEDED', 'FAILED', 'SKIPPED')),
    error              TEXT,
    event_count        INTEGER NOT NULL,
    superseded_at      TEXT,
    projection_version INTEGER
);

DROP INDEX IF EXISTS uq_run_dedupe;

CREATE UNIQUE INDEX IF NOT EXISTS uq_run_dedupe_active
    ON extraction_runs(source_turn_id, extractor_version)
    WHERE status = 'SUCCEEDED' AND superseded_at IS NULL;

CREATE TABLE IF NOT EXISTS superseded_runs (
    old_run_id      TEXT PRIMARY KEY,
    new_run_id      TEXT NOT NULL,
    superseded_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id                  TEXT PRIMARY KEY,
    seq                 INTEGER NOT NULL UNIQUE,
    observed_at         TEXT NOT NULL,
    effective_at_start  TEXT,
    effective_at_end    TEXT,
    recorded_at         TEXT NOT NULL,
    type                TEXT NOT NULL,
    data                TEXT NOT NULL CHECK(json_valid(data)),
    extraction_run_id   TEXT,
    source_turn_id      TEXT,
    source_role         TEXT NOT NULL,
    confidence          REAL,
    reason              TEXT,
    time_confidence     TEXT NOT NULL CHECK(time_confidence IN ('exact', 'inferred', 'unknown')),
    caused_by           TEXT,
    schema_version      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_recorded_at ON events(recorded_at);
CREATE INDEX IF NOT EXISTS idx_events_effective_start ON events(effective_at_start);
CREATE INDEX IF NOT EXISTS idx_events_effective_end ON events(effective_at_end);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(extraction_run_id);

CREATE TABLE IF NOT EXISTS event_entities (
    event_id     TEXT NOT NULL,
    entity_id    TEXT NOT NULL,
    role         TEXT NOT NULL,
    PRIMARY KEY(event_id, entity_id, role)
);

CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);

CREATE TABLE IF NOT EXISTS dirty_ranges (
    id                 TEXT PRIMARY KEY,
    owner_id           TEXT NOT NULL,
    from_recorded_at   TEXT,
    from_effective_at  TEXT,
    reason             TEXT NOT NULL,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    id                         TEXT PRIMARY KEY,
    basis                      TEXT NOT NULL CHECK(basis IN ('known')),
    created_at                 TEXT NOT NULL,
    last_seq                   INTEGER NOT NULL,
    projection_version         INTEGER NOT NULL,
    max_recorded_at_included   TEXT NOT NULL,
    max_effective_at_included  TEXT,
    state_blob                 BLOB NOT NULL,
    relation_blob              BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS vec_events (
    event_id           TEXT NOT NULL,
    embedder_version   TEXT NOT NULL,
    dim                INTEGER NOT NULL,
    embedding          BLOB NOT NULL,
    indexed_at         TEXT NOT NULL,
    PRIMARY KEY(event_id, embedder_version)
);

CREATE INDEX IF NOT EXISTS idx_vec_events_version ON vec_events(embedder_version);
