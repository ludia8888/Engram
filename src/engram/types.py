from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


SourceRole = Literal["user", "assistant", "tool", "system", "manual"]
TimeConfidence = Literal["exact", "inferred", "unknown"]
ExtractionRunStatus = Literal["SUCCEEDED", "FAILED", "SKIPPED"]
RelationDirection = Literal["outgoing", "incoming"]


@dataclass(slots=True)
class TurnAck:
    turn_id: str
    observed_at: datetime
    durable_at: datetime
    queued: bool


@dataclass(slots=True)
class RawTurn:
    id: str
    session_id: str | None
    observed_at: datetime
    user: str
    assistant: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueueItem:
    turn_id: str
    observed_at: datetime
    session_id: str | None
    user: str
    assistant: str
    metadata: dict[str, Any]

    @classmethod
    def from_turn(cls, turn: RawTurn) -> "QueueItem":
        return cls(
            turn_id=turn.id,
            observed_at=turn.observed_at,
            session_id=turn.session_id,
            user=turn.user,
            assistant=turn.assistant,
            metadata=dict(turn.metadata),
        )


@dataclass(slots=True)
class Event:
    id: str
    seq: int
    observed_at: datetime
    effective_at_start: datetime | None
    effective_at_end: datetime | None
    recorded_at: datetime
    type: str
    data: dict[str, Any]
    extraction_run_id: str | None
    source_turn_id: str | None
    source_role: SourceRole
    confidence: float | None
    reason: str | None
    time_confidence: TimeConfidence
    caused_by: str | None
    schema_version: int


@dataclass(slots=True)
class ExtractedEvent:
    type: str
    data: dict[str, Any]
    effective_at_start: datetime | None = None
    effective_at_end: datetime | None = None
    caused_by: str | None = None
    source_role: SourceRole = "user"
    confidence: float | None = None
    reason: str | None = None
    time_confidence: TimeConfidence = "unknown"


@dataclass(slots=True)
class ExtractionRun:
    id: str
    source_turn_id: str
    extractor_version: str
    observed_at: datetime
    processed_at: datetime
    status: ExtractionRunStatus
    error: str | None
    event_count: int
    superseded_at: datetime | None
    projection_version: int | None


@dataclass(slots=True)
class Entity:
    id: str
    type: str
    attrs: dict[str, Any]
    created_recorded_at: datetime
    updated_recorded_at: datetime


@dataclass(slots=True)
class TemporalEntityView:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    unknown_attrs: list[str]
    supporting_event_ids: list[str]
    basis: Literal["known", "valid"]
    as_of: datetime


@dataclass(slots=True)
class SearchResult:
    entity_id: str
    score: float
    matched_axes: set[Literal["entity", "semantic", "temporal", "causal"]]
    supporting_event_ids: list[str]
    time_basis: Literal["known", "valid"]


@dataclass(slots=True)
class HistoryEntry:
    entity_id: str
    attr: str
    old_value: Any
    new_value: Any
    observed_at: datetime
    effective_at_start: datetime | None
    effective_at_end: datetime | None
    recorded_at: datetime
    reason: str | None
    confidence: float | None
    basis: Literal["known", "valid"]
    event_id: str


@dataclass(slots=True)
class RelationHistoryEntry:
    entity_id: str
    other_entity_id: str
    relation_type: str
    direction: RelationDirection
    action: Literal["create", "update", "delete"]
    attrs: dict[str, Any]
    observed_at: datetime
    effective_at_start: datetime | None
    effective_at_end: datetime | None
    recorded_at: datetime
    reason: str | None
    confidence: float | None
    basis: Literal["known", "valid"]
    event_id: str


@dataclass(frozen=True, slots=True)
class RelationEdge:
    relation_type: str
    other_entity_id: str
    direction: RelationDirection
    attrs: dict[str, Any]
