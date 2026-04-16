from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response

from engram.time_utils import from_rfc3339

from .models import (
    AppendRequest,
    AppendResponse,
    DuplicateCandidateResponse,
    EntityResponse,
    FlushRequest,
    HealthResponse,
    HistoryEntryResponse,
    ProjectionRebuildResultResponse,
    RawTurnResponse,
    RebuildProjectionRequest,
    MergeEntitiesRequest,
    MergeEntitiesResponse,
    RelationEdgeResponse,
    RelationHistoryEntryResponse,
    ReprocessRequest,
    ReprocessResponse,
    SearchResultResponse,
    TemporalEntityViewResponse,
    TurnAckResponse,
    TurnRequest,
)

router = APIRouter()


def _engram(request: Request):
    return request.app.state.engram


def _parse_dt(value: str | None, name: str):
    if value is None:
        return None
    try:
        return from_rfc3339(value)
    except Exception:
        raise HTTPException(400, detail=f"Invalid datetime for {name}: {value}")


def _parse_time_window(start: str | None, end: str | None):
    tw_start = _parse_dt(start, "time_window_start")
    tw_end = _parse_dt(end, "time_window_end")
    if (tw_start is None) != (tw_end is None):
        raise HTTPException(400, detail="time_window_start and time_window_end must both be provided or both omitted")
    if tw_start is not None and tw_end is not None and tw_start >= tw_end:
        raise HTTPException(400, detail="time_window_start must be before time_window_end")
    return (tw_start, tw_end) if tw_start and tw_end else None


@router.get("/health")
def health(request: Request) -> HealthResponse:
    mem = _engram(request)
    config = request.app.state.config
    worker = getattr(mem, "_background_worker", None)
    return HealthResponse(
        user_id=config["user_id"],
        auto_flush=config["auto_flush"],
        worker_alive=worker.is_alive if worker is not None else None,
    )


@router.post("/turn", status_code=201)
def create_turn(request: Request, body: TurnRequest) -> TurnAckResponse:
    mem = _engram(request)
    ack = mem.turn(
        user=body.user,
        assistant=body.assistant,
        observed_at=body.observed_at,
        session_id=body.session_id,
        metadata=body.metadata,
    )
    return TurnAckResponse.model_validate(ack)


@router.post("/append", status_code=201)
def append_event(request: Request, body: AppendRequest) -> AppendResponse:
    mem = _engram(request)
    event_id = mem.append(
        event_type=body.event_type,
        data=body.data,
        observed_at=body.observed_at,
        effective_at_start=body.effective_at_start,
        effective_at_end=body.effective_at_end,
        source_role=body.source_role,
        source_turn_id=body.source_turn_id,
        caused_by=body.caused_by,
        confidence=body.confidence,
        reason=body.reason,
        time_confidence=body.time_confidence,
    )
    return AppendResponse(event_id=event_id)


@router.post("/entities/merge")
def merge_entities(request: Request, body: MergeEntitiesRequest) -> MergeEntitiesResponse:
    mem = _engram(request)
    merged_to = mem.merge_entities(body.source_id, body.target_id, reason=body.reason)
    return MergeEntitiesResponse(merged_to=merged_to)


@router.get("/duplicates")
def list_duplicates(
    request: Request,
    entity_id: str | None = None,
    status: str | None = "OPEN",
    limit: int = Query(default=100, ge=1, le=500),
) -> list[DuplicateCandidateResponse]:
    mem = _engram(request)
    rows = mem.list_duplicate_candidates(entity_id=entity_id, status=status, limit=limit)
    return [DuplicateCandidateResponse.model_validate(row) for row in rows]


@router.get("/entity/{entity_id:path}/known-at")
def get_known_at(
    request: Request,
    entity_id: str,
    at: str = Query(...),
) -> TemporalEntityViewResponse:
    mem = _engram(request)
    target = _parse_dt(at, "at")
    view = mem.get_known_at(entity_id, target)
    if view is None:
        raise HTTPException(404, detail=f"Entity not found: {entity_id}")
    return TemporalEntityViewResponse.model_validate(view)


@router.get("/entity/{entity_id:path}/valid-at")
def get_valid_at(
    request: Request,
    entity_id: str,
    at: str = Query(...),
) -> TemporalEntityViewResponse:
    mem = _engram(request)
    target = _parse_dt(at, "at")
    view = mem.get_valid_at(entity_id, target)
    if view is None:
        raise HTTPException(404, detail=f"Entity not found: {entity_id}")
    return TemporalEntityViewResponse.model_validate(view)


@router.get("/entity/{entity_id:path}/history")
def get_history(
    request: Request,
    entity_id: str,
    attr: str | None = None,
    time_mode: Literal["known", "valid"] = "known",
) -> list[HistoryEntryResponse]:
    mem = _engram(request)
    if time_mode == "known":
        entries = mem.known_history(entity_id, attr=attr)
    else:
        entries = mem.valid_history(entity_id, attr=attr)
    return [HistoryEntryResponse.model_validate(e) for e in entries]


@router.get("/entity/{entity_id:path}/relations")
def get_relations(
    request: Request,
    entity_id: str,
    time_mode: Literal["known", "valid"] = "known",
    at: str | None = None,
    time_window_start: str | None = None,
    time_window_end: str | None = None,
) -> list[RelationEdgeResponse]:
    mem = _engram(request)
    at_dt = _parse_dt(at, "at")
    time_window = _parse_time_window(time_window_start, time_window_end)
    edges = mem.get_relations(entity_id, time_mode=time_mode, at=at_dt, time_window=time_window)
    return [RelationEdgeResponse.model_validate(e) for e in edges]


@router.get("/search")
def search(
    request: Request,
    query: str = Query(..., min_length=1),
    time_mode: Literal["known", "valid"] = "known",
    time_window_start: str | None = None,
    time_window_end: str | None = None,
    k: int = Query(default=20, ge=1),
) -> list[SearchResultResponse]:
    mem = _engram(request)
    time_window = _parse_time_window(time_window_start, time_window_end)
    results = mem.search(query, time_mode=time_mode, time_window=time_window, k=k)
    return [SearchResultResponse.model_validate(r) for r in results]


@router.get("/context")
def context(
    request: Request,
    query: str = Query(..., min_length=1),
    time_mode: Literal["known", "valid"] = "known",
    time_window_start: str | None = None,
    time_window_end: str | None = None,
    max_tokens: int = Query(default=2000, ge=1),
    include_history: bool = True,
    include_raw: bool = False,
):
    mem = _engram(request)
    time_window = _parse_time_window(time_window_start, time_window_end)
    text = mem.context(
        query,
        time_mode=time_mode,
        time_window=time_window,
        max_tokens=max_tokens,
        include_history=include_history,
        include_raw=include_raw,
    )
    return Response(content=text, media_type="text/plain")


@router.get("/entity/{entity_id:path}/relation-history")
def relation_history(
    request: Request,
    entity_id: str,
    relation_type: str | None = None,
    other_entity_id: str | None = None,
    time_mode: Literal["known", "valid"] = "known",
) -> list[RelationHistoryEntryResponse]:
    mem = _engram(request)
    entries = mem.relation_history(
        entity_id,
        relation_type=relation_type,
        other_entity_id=other_entity_id,
        time_mode=time_mode,
    )
    return [RelationHistoryEntryResponse.model_validate(e) for e in entries]


@router.get("/entity/{entity_id:path}")
def get_entity(request: Request, entity_id: str) -> EntityResponse:
    mem = _engram(request)
    entity = mem.get(entity_id)
    if entity is None:
        raise HTTPException(404, detail=f"Entity not found: {entity_id}")
    return EntityResponse.model_validate(entity)


@router.post("/reprocess")
def reprocess(request: Request, body: ReprocessRequest) -> ReprocessResponse:
    mem = _engram(request)
    count = mem.reprocess(
        from_turn_id=body.from_turn_id,
        to_turn_id=body.to_turn_id,
        extractor_version=body.extractor_version,
    )
    return ReprocessResponse(count=count)


@router.post("/rebuild-projection")
def rebuild_projection(request: Request, body: RebuildProjectionRequest) -> ProjectionRebuildResultResponse:
    mem = _engram(request)
    result = mem.rebuild_projection(owner_id=body.owner_id, mode=body.mode)
    return ProjectionRebuildResultResponse.model_validate(result)


@router.post("/flush")
def flush(request: Request, body: FlushRequest):
    mem = _engram(request)
    mem.flush(body.level)
    return Response(status_code=204)


@router.get("/raw/{turn_id}")
def raw_get(request: Request, turn_id: str) -> RawTurnResponse:
    mem = _engram(request)
    turn = mem.raw_get(turn_id)
    if turn is None:
        raise HTTPException(404, detail=f"Turn not found: {turn_id}")
    return RawTurnResponse.model_validate(turn)
