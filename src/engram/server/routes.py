from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response

from engram.time_utils import from_rfc3339

from .models import (
    AppendRequest,
    AppendResponse,
    EntityResponse,
    FlushRequest,
    HealthResponse,
    HistoryEntryResponse,
    RawTurnResponse,
    RelationEdgeResponse,
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


@router.post("/turn")
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


@router.post("/append")
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
    tw_start = _parse_dt(time_window_start, "time_window_start")
    tw_end = _parse_dt(time_window_end, "time_window_end")
    time_window = (tw_start, tw_end) if tw_start and tw_end else None
    if (tw_start is None) != (tw_end is None):
        raise HTTPException(400, detail="time_window_start and time_window_end must both be provided or both omitted")
    edges = mem.get_relations(entity_id, time_mode=time_mode, at=at_dt, time_window=time_window)
    return [RelationEdgeResponse.model_validate(e) for e in edges]


@router.get("/entity/{entity_id:path}")
def get_entity(request: Request, entity_id: str) -> EntityResponse:
    mem = _engram(request)
    entity = mem.get(entity_id)
    if entity is None:
        raise HTTPException(404, detail=f"Entity not found: {entity_id}")
    return EntityResponse.model_validate(entity)


@router.get("/search")
def search(
    request: Request,
    query: str = Query(..., min_length=1),
    time_mode: Literal["known", "valid"] = "known",
    time_window_start: str | None = None,
    time_window_end: str | None = None,
    k: int = 20,
) -> list[SearchResultResponse]:
    mem = _engram(request)
    tw_start = _parse_dt(time_window_start, "time_window_start")
    tw_end = _parse_dt(time_window_end, "time_window_end")
    time_window = (tw_start, tw_end) if tw_start and tw_end else None
    if (tw_start is None) != (tw_end is None):
        raise HTTPException(400, detail="time_window_start and time_window_end must both be provided or both omitted")
    results = mem.search(query, time_mode=time_mode, time_window=time_window, k=k)
    return [SearchResultResponse.model_validate(r) for r in results]


@router.get("/context")
def context(
    request: Request,
    query: str = Query(..., min_length=1),
    time_mode: Literal["known", "valid"] = "known",
    time_window_start: str | None = None,
    time_window_end: str | None = None,
    max_tokens: int = 2000,
    include_history: bool = True,
    include_raw: bool = False,
):
    mem = _engram(request)
    tw_start = _parse_dt(time_window_start, "time_window_start")
    tw_end = _parse_dt(time_window_end, "time_window_end")
    time_window = (tw_start, tw_end) if tw_start and tw_end else None
    if (tw_start is None) != (tw_end is None):
        raise HTTPException(400, detail="time_window_start and time_window_end must both be provided or both omitted")
    text = mem.context(
        query,
        time_mode=time_mode,
        time_window=time_window,
        max_tokens=max_tokens,
        include_history=include_history,
        include_raw=include_raw,
    )
    return Response(content=text, media_type="text/plain")


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
