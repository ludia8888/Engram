from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, PlainSerializer, field_validator

from engram.time_utils import to_rfc3339

RFC3339DateTime = Annotated[datetime, PlainSerializer(to_rfc3339, return_type=str)]


class TurnRequest(BaseModel):
    user: str
    assistant: str
    observed_at: datetime | None = None
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class AppendRequest(BaseModel):
    event_type: str
    data: dict[str, Any]
    observed_at: datetime | None = None
    effective_at_start: datetime | None = None
    effective_at_end: datetime | None = None
    source_role: str = "manual"
    source_turn_id: str | None = None
    caused_by: str | None = None
    confidence: float | None = None
    reason: str | None = None
    time_confidence: str = "unknown"


class FlushRequest(BaseModel):
    level: Literal["raw", "canonical", "projection", "snapshot", "index", "all"] = "projection"


class TurnAckResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    turn_id: str
    observed_at: RFC3339DateTime
    durable_at: RFC3339DateTime
    queued: bool


class EntityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    type: str
    attrs: dict[str, Any]
    created_recorded_at: RFC3339DateTime
    updated_recorded_at: RFC3339DateTime


class TemporalEntityViewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    unknown_attrs: list[str]
    supporting_event_ids: list[str]
    basis: Literal["known", "valid"]
    as_of: RFC3339DateTime


class HistoryEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    entity_id: str
    attr: str
    old_value: Any
    new_value: Any
    observed_at: RFC3339DateTime
    effective_at_start: RFC3339DateTime | None
    effective_at_end: RFC3339DateTime | None
    recorded_at: RFC3339DateTime
    reason: str | None
    confidence: float | None
    basis: Literal["known", "valid"]
    event_id: str


class RelationEdgeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    relation_type: str
    other_entity_id: str
    direction: Literal["outgoing", "incoming"]
    attrs: dict[str, Any]


class SearchResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    entity_id: str
    score: float
    matched_axes: list[str]
    supporting_event_ids: list[str]
    time_basis: Literal["known", "valid"]

    @field_validator("matched_axes", mode="before")
    @classmethod
    def _coerce_matched_axes(cls, v):
        if isinstance(v, set):
            return sorted(v)
        return v


class RawTurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str | None
    observed_at: RFC3339DateTime
    user: str
    assistant: str
    metadata: dict[str, Any]


class AppendResponse(BaseModel):
    event_id: str


class RelationHistoryEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    entity_id: str
    other_entity_id: str
    relation_type: str
    direction: Literal["outgoing", "incoming"]
    action: Literal["create", "update", "delete"]
    attrs: dict[str, Any]
    observed_at: RFC3339DateTime
    effective_at_start: RFC3339DateTime | None
    effective_at_end: RFC3339DateTime | None
    recorded_at: RFC3339DateTime
    reason: str | None
    confidence: float | None
    basis: Literal["known", "valid"]
    event_id: str


class ProjectionRebuildResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    scope: Literal["dirty", "owner", "full"]
    target_owner_id: str | None
    rebuilt_owner_count: int
    dirty_owner_count_before: int
    dirty_owner_count_after: int


class ReprocessRequest(BaseModel):
    from_turn_id: str | None = None
    to_turn_id: str | None = None
    extractor_version: str | None = None


class RebuildProjectionRequest(BaseModel):
    owner_id: str | None = None
    mode: Literal["dirty", "full"] = "dirty"


class ReprocessResponse(BaseModel):
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    user_id: str
    auto_flush: bool
    worker_alive: bool | None = None
