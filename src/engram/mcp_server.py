from __future__ import annotations

import json
import os
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from .canonical import NullExtractor
from .engram import Engram
from .meaning_index import NullMeaningAnalyzer
from .semantic import HashEmbedder
from .time_utils import from_rfc3339, to_rfc3339

mcp = FastMCP(
    "engram",
    instructions=(
        "Engram is a structured long-term memory engine for LLM agents. "
        "Use engram_turn to store conversations, engram_append to record structured observations, "
        "engram_recall to get relevant memory context for a query, "
        "engram_get to look up a specific entity, and engram_search for retrieval."
    ),
)

_engram: Engram | None = None


def _get_engram() -> Engram:
    global _engram
    if _engram is None:
        _engram = Engram(
            user_id=os.environ.get("ENGRAM_USER_ID", "default"),
            path=os.environ.get("ENGRAM_PATH"),
            session_id=os.environ.get("ENGRAM_SESSION_ID"),
            extractor=_build_extractor(),
            embedder=_build_embedder(),
            meaning_analyzer=_build_meaning_analyzer(),
            auto_flush=os.environ.get("ENGRAM_AUTO_FLUSH", "true").lower() in ("true", "1", "yes"),
        )
    return _engram


def _build_extractor():
    name = os.environ.get("ENGRAM_EXTRACTOR", "null")
    if name == "null":
        return NullExtractor()
    if name == "openai":
        from .openai_extractor import OpenAIExtractor

        return OpenAIExtractor(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_MODEL", "gpt-5.4-mini"),
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown extractor: {name}")


def _build_embedder():
    name = os.environ.get("ENGRAM_EMBEDDER", "hash")
    if name == "hash":
        return HashEmbedder()
    if name == "openai":
        from .semantic import OpenAIEmbedder

        dims = os.environ.get("ENGRAM_OPENAI_EMBED_DIMS")
        return OpenAIEmbedder(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            dimensions=int(dims) if dims else None,
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown embedder: {name}")


def _build_meaning_analyzer():
    name = os.environ.get("ENGRAM_MEANING_ANALYZER", "null")
    if name == "null":
        return NullMeaningAnalyzer()
    if name == "openai":
        from .openai_meaning_analyzer import OpenAIMeaningAnalyzer

        return OpenAIMeaningAnalyzer(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_MEANING_MODEL", "gpt-5.4-mini"),
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown meaning analyzer: {name}")


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return from_rfc3339(value)


def _entity_to_dict(entity) -> dict:
    return {
        "id": entity.id,
        "type": entity.type,
        "attrs": entity.attrs,
        "created_recorded_at": to_rfc3339(entity.created_recorded_at),
        "updated_recorded_at": to_rfc3339(entity.updated_recorded_at),
    }


def _view_to_dict(view) -> dict:
    return {
        "entity_id": view.entity_id,
        "entity_type": view.entity_type,
        "attrs": view.attrs,
        "unknown_attrs": view.unknown_attrs,
        "basis": view.basis,
        "as_of": to_rfc3339(view.as_of),
    }


@mcp.tool()
def engram_turn(
    user: str,
    assistant: str,
    observed_at: str | None = None,
    session_id: str | None = None,
) -> str:
    """Store a conversation turn in memory. The engine automatically extracts
    entities, relations, and temporal facts from the dialogue.

    Args:
        user: What the user said
        assistant: What the assistant replied
        observed_at: Optional ISO 8601 UTC timestamp (defaults to now)
        session_id: Optional session identifier for grouping turns
    """
    mem = _get_engram()
    ack = mem.turn(
        user=user,
        assistant=assistant,
        observed_at=_parse_dt(observed_at),
        session_id=session_id,
    )
    return json.dumps({
        "turn_id": ack.turn_id,
        "observed_at": to_rfc3339(ack.observed_at),
        "durable_at": to_rfc3339(ack.durable_at),
        "queued": ack.queued,
    })


@mcp.tool()
def engram_append(
    event_type: str,
    data: str,
    observed_at: str | None = None,
    effective_at_start: str | None = None,
    effective_at_end: str | None = None,
    caused_by: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
) -> str:
    """Record a structured observation or decision directly as a memory event.
    Use this when the agent already has structured data (findings, decisions, corrections).

    Args:
        event_type: One of entity.create, entity.update, entity.delete, relation.create, relation.update, relation.delete
        data: JSON string with event payload (e.g. {"id": "finding:bug-1", "type": "finding", "attrs": {"severity": "high"}})
        observed_at: Optional ISO 8601 UTC timestamp
        effective_at_start: Optional ISO 8601 UTC — when this fact became true
        effective_at_end: Optional ISO 8601 UTC — when this fact stopped being true
        caused_by: Optional event_id that caused this observation
        confidence: Optional float 0.0-1.0
        reason: Optional explanation
    """
    mem = _get_engram()
    parsed_data = json.loads(data) if isinstance(data, str) else data
    event_id = mem.append(
        event_type=event_type,
        data=parsed_data,
        observed_at=_parse_dt(observed_at),
        effective_at_start=_parse_dt(effective_at_start),
        effective_at_end=_parse_dt(effective_at_end),
        caused_by=caused_by,
        confidence=confidence,
        reason=reason,
    )
    return json.dumps({"event_id": event_id})


@mcp.tool()
def engram_recall(
    query: str,
    time_mode: str = "known",
    max_tokens: int = 2000,
    include_history: bool = True,
    include_raw: bool = False,
) -> str:
    """Retrieve relevant memory context for a query. Returns a structured text
    block ready to be used as LLM system prompt context.

    This is the primary tool for an agent to recall past knowledge.

    Args:
        query: Natural language query (e.g. "앨리스의 식단과 위치")
        time_mode: "known" (what was known at the time) or "valid" (what was actually true)
        max_tokens: Maximum tokens for the context block
        include_history: Include change history in context
        include_raw: Include raw conversation evidence
    """
    mem = _get_engram()
    return mem.context(
        query,
        time_mode=time_mode,
        max_tokens=max_tokens,
        include_history=include_history,
        include_raw=include_raw,
    )


@mcp.tool()
def engram_get(entity_id: str) -> str:
    """Look up the current state of a specific entity by ID.

    Args:
        entity_id: Entity identifier (e.g. "user:alice", "finding:bug-1")
    """
    mem = _get_engram()
    entity = mem.get(entity_id)
    if entity is None:
        return json.dumps({"error": f"Entity not found: {entity_id}"})
    return json.dumps(_entity_to_dict(entity))


@mcp.tool()
def engram_search(
    query: str,
    k: int = 10,
    time_mode: str = "known",
) -> str:
    """Search memory for entities matching a query. Returns ranked results
    with scores and matched axes (entity, semantic, meaning, causal, temporal).

    Args:
        query: Search query
        k: Maximum number of results
        time_mode: "known" or "valid"
    """
    mem = _get_engram()
    results = mem.search(query, k=k, time_mode=time_mode)
    return json.dumps([
        {
            "entity_id": r.entity_id,
            "score": r.score,
            "matched_axes": sorted(r.matched_axes),
            "time_basis": r.time_basis,
        }
        for r in results
    ])


@mcp.tool()
def engram_get_relations(entity_id: str, time_mode: str = "known") -> str:
    """Get all relations for an entity (e.g. who is their manager, what project are they on).

    Args:
        entity_id: Entity identifier
        time_mode: "known" or "valid"
    """
    mem = _get_engram()
    edges = mem.get_relations(entity_id, time_mode=time_mode)
    return json.dumps([
        {
            "relation_type": e.relation_type,
            "other_entity_id": e.other_entity_id,
            "direction": e.direction,
            "attrs": e.attrs,
        }
        for e in edges
    ])


@mcp.tool()
def engram_history(
    entity_id: str,
    attr: str | None = None,
    time_mode: str = "known",
) -> str:
    """Get the change history for an entity's attributes.
    Shows what changed, when, why, and with what confidence.

    Args:
        entity_id: Entity identifier
        attr: Optional specific attribute to filter (e.g. "location")
        time_mode: "known" or "valid"
    """
    mem = _get_engram()
    if time_mode == "valid":
        entries = mem.valid_history(entity_id, attr=attr)
    else:
        entries = mem.known_history(entity_id, attr=attr)
    return json.dumps([
        {
            "attr": e.attr,
            "old_value": e.old_value,
            "new_value": e.new_value,
            "recorded_at": to_rfc3339(e.recorded_at),
            "effective_at_start": to_rfc3339(e.effective_at_start) if e.effective_at_start else None,
            "reason": e.reason,
            "confidence": e.confidence,
            "basis": e.basis,
        }
        for e in entries
    ])


@mcp.tool()
def engram_flush(level: str = "all") -> str:
    """Flush the memory pipeline. Use after batch operations or when you need
    immediate consistency.

    Args:
        level: "canonical" (extract events), "projection" (rebuild state),
               "snapshot" (persist), "index" (reindex), "all" (full pipeline)
    """
    mem = _get_engram()
    mem.flush(level)
    return json.dumps({"status": "ok", "level": level})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
