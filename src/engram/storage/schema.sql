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
CREATE INDEX IF NOT EXISTS idx_event_entities_entity_event ON event_entities(entity_id, event_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id           TEXT NOT NULL,
    entity_type         TEXT NOT NULL,
    alias               TEXT NOT NULL,
    normalized_alias    TEXT NOT NULL,
    alias_kind          TEXT NOT NULL CHECK(alias_kind IN ('name', 'alias', 'canonical_key', 'id')),
    created_at          TEXT NOT NULL,
    PRIMARY KEY(entity_id, alias_kind, normalized_alias)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_lookup
    ON entity_aliases(entity_type, alias_kind, normalized_alias, entity_id);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_lookup_any_kind
    ON entity_aliases(entity_type, normalized_alias, entity_id);

CREATE TABLE IF NOT EXISTS entity_redirects (
    source_entity_id    TEXT PRIMARY KEY,
    target_entity_id    TEXT NOT NULL,
    merged_at           TEXT NOT NULL,
    reason              TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_redirects_target
    ON entity_redirects(target_entity_id);

CREATE TABLE IF NOT EXISTS duplicate_candidates (
    id                  TEXT PRIMARY KEY,
    entity_id           TEXT NOT NULL,
    candidate_entity_id TEXT NOT NULL,
    match_basis         TEXT NOT NULL,
    score               REAL NOT NULL,
    status              TEXT NOT NULL CHECK(status IN ('OPEN', 'MERGED', 'DISMISSED')),
    reason              TEXT,
    observed_at         TEXT NOT NULL,
    source_turn_id      TEXT,
    event_type          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_duplicate_candidates_entity
    ON duplicate_candidates(entity_id, status, score DESC, candidate_entity_id);

CREATE INDEX IF NOT EXISTS idx_duplicate_candidates_candidate
    ON duplicate_candidates(candidate_entity_id, status, score DESC, entity_id);

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
CREATE INDEX IF NOT EXISTS idx_vec_events_version_event ON vec_events(embedder_version, event_id);

CREATE TABLE IF NOT EXISTS event_search_terms (
    event_id     TEXT NOT NULL,
    term         TEXT NOT NULL,
    PRIMARY KEY(event_id, term)
);

CREATE INDEX IF NOT EXISTS idx_event_search_terms_term ON event_search_terms(term);

CREATE TABLE IF NOT EXISTS event_search_units (
    event_id           TEXT NOT NULL,
    analyzer_version   TEXT NOT NULL,
    unit_kind          TEXT NOT NULL CHECK(unit_kind IN ('protected_phrase', 'alias', 'canonical_key', 'facet', 'fallback_term')),
    unit_key           TEXT NOT NULL DEFAULT '',
    unit_value         TEXT NOT NULL,
    normalized_value   TEXT NOT NULL,
    confidence         REAL,
    metadata           TEXT CHECK(metadata IS NULL OR json_valid(metadata)),
    PRIMARY KEY(event_id, analyzer_version, unit_kind, unit_key, normalized_value)
);

CREATE INDEX IF NOT EXISTS idx_event_search_units_lookup
    ON event_search_units(analyzer_version, unit_kind, normalized_value);

CREATE INDEX IF NOT EXISTS idx_event_search_units_facet_lookup
    ON event_search_units(analyzer_version, unit_kind, unit_key, normalized_value);

CREATE INDEX IF NOT EXISTS idx_event_search_units_event
    ON event_search_units(event_id, analyzer_version);

CREATE TABLE IF NOT EXISTS meaning_analysis_runs (
    event_id           TEXT NOT NULL,
    analyzer_version   TEXT NOT NULL,
    processed_at       TEXT NOT NULL,
    status             TEXT NOT NULL CHECK(status IN ('SUCCEEDED', 'FAILED', 'SKIPPED')),
    error              TEXT,
    unit_count         INTEGER NOT NULL,
    PRIMARY KEY(event_id, analyzer_version)
);

CREATE INDEX IF NOT EXISTS idx_meaning_analysis_runs_version_status
    ON meaning_analysis_runs(analyzer_version, status);

CREATE TABLE IF NOT EXISTS query_meaning_cache (
    normalized_query   TEXT NOT NULL,
    analyzer_version   TEXT NOT NULL,
    payload            TEXT NOT NULL CHECK(json_valid(payload)),
    cached_at          TEXT NOT NULL,
    PRIMARY KEY(normalized_query, analyzer_version)
);

CREATE INDEX IF NOT EXISTS idx_query_meaning_cache_version_cached_at
    ON query_meaning_cache(analyzer_version, cached_at DESC, normalized_query ASC);
