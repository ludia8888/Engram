from __future__ import annotations

import argparse
import json
import os
import sys

from .config import build_embedder, build_extractor, build_meaning_analyzer
from .engram import Engram
from .time_utils import to_rfc3339


def _build_engram(args) -> Engram:
    return Engram(
        user_id=args.user_id,
        path=args.path,
        extractor=build_extractor(),
        embedder=build_embedder(),
        meaning_analyzer=build_meaning_analyzer(),
        auto_flush=False,
    )


def cmd_turn(args) -> None:
    mem = _build_engram(args)
    ack = mem.turn(user=args.user, assistant=args.assistant)
    mem.flush("all")
    print(json.dumps({
        "turn_id": ack.turn_id,
        "observed_at": to_rfc3339(ack.observed_at),
        "queued": ack.queued,
    }, indent=2))
    mem.close()


def cmd_append(args) -> None:
    mem = _build_engram(args)
    data = json.loads(args.data)
    event_id = mem.append(
        event_type=args.event_type,
        data=data,
        reason=args.reason,
    )
    mem.flush("all")
    print(json.dumps({"event_id": event_id}, indent=2))
    mem.close()


def cmd_get(args) -> None:
    mem = _build_engram(args)
    entity = mem.get(args.entity_id)
    if entity is None:
        print(f"Entity not found: {args.entity_id}", file=sys.stderr)
        mem.close()
        sys.exit(1)
    print(json.dumps({
        "id": entity.id,
        "type": entity.type,
        "attrs": entity.attrs,
        "created_recorded_at": to_rfc3339(entity.created_recorded_at),
        "updated_recorded_at": to_rfc3339(entity.updated_recorded_at),
        "redirected_from": entity.redirected_from,
    }, ensure_ascii=False, indent=2))
    mem.close()


def cmd_search(args) -> None:
    mem = _build_engram(args)
    results = mem.search(args.query, k=args.k, time_mode=args.time_mode)
    output = [
        {
            "entity_id": r.entity_id,
            "score": r.score,
            "matched_axes": sorted(r.matched_axes),
        }
        for r in results
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    mem.close()


def cmd_context(args) -> None:
    mem = _build_engram(args)
    text = mem.context(
        args.query,
        time_mode=args.time_mode,
        max_tokens=args.max_tokens,
    )
    print(text)
    mem.close()


def cmd_history(args) -> None:
    mem = _build_engram(args)
    entries = mem.known_history(args.entity_id, attr=args.attr)
    output = [
        {
            "attr": e.attr,
            "old_value": e.old_value,
            "new_value": e.new_value,
            "recorded_at": to_rfc3339(e.recorded_at),
            "reason": e.reason,
            "confidence": e.confidence,
        }
        for e in entries
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    mem.close()


def cmd_flush(args) -> None:
    mem = _build_engram(args)
    mem.flush(args.level)
    print(f"Flushed: {args.level}")
    mem.close()


def cmd_merge(args) -> None:
    mem = _build_engram(args)
    merged_to = mem.merge_entities(args.source_id, args.target_id, reason=args.reason)
    print(json.dumps({"merged_to": merged_to}, ensure_ascii=False, indent=2))
    mem.close()


def cmd_duplicates(args) -> None:
    mem = _build_engram(args)
    rows = mem.list_duplicate_candidates(entity_id=args.entity_id, status=args.status, limit=args.limit)
    print(
        json.dumps(
            [
                {
                    "id": row.id,
                    "entity_id": row.entity_id,
                    "candidate_entity_id": row.candidate_entity_id,
                    "match_basis": row.match_basis,
                    "score": row.score,
                    "status": row.status,
                    "reason": row.reason,
                    "observed_at": to_rfc3339(row.observed_at),
                    "source_turn_id": row.source_turn_id,
                    "event_type": row.event_type,
                }
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    mem.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="engram", description="Engram memory engine CLI")
    parser.add_argument("--user-id", default=os.environ.get("ENGRAM_USER_ID", "default"))
    parser.add_argument("--path", default=os.environ.get("ENGRAM_PATH"))

    sub = parser.add_subparsers(dest="command", required=True)

    p_turn = sub.add_parser("turn", help="Store a conversation turn")
    p_turn.add_argument("--user", required=True)
    p_turn.add_argument("--assistant", required=True)
    p_turn.set_defaults(func=cmd_turn)

    p_append = sub.add_parser("append", help="Append a structured event")
    p_append.add_argument("event_type", help="e.g. entity.create, entity.update")
    p_append.add_argument("data", help="JSON string for event data")
    p_append.add_argument("--reason", default=None)
    p_append.set_defaults(func=cmd_append)

    p_get = sub.add_parser("get", help="Get entity current state")
    p_get.add_argument("entity_id")
    p_get.set_defaults(func=cmd_get)

    p_search = sub.add_parser("search", help="Search memory")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=10)
    p_search.add_argument("--time-mode", default="known", choices=["known", "valid"])
    p_search.set_defaults(func=cmd_search)

    p_context = sub.add_parser("context", help="Generate LLM context from memory")
    p_context.add_argument("query")
    p_context.add_argument("--time-mode", default="known", choices=["known", "valid"])
    p_context.add_argument("--max-tokens", type=int, default=2000)
    p_context.set_defaults(func=cmd_context)

    p_history = sub.add_parser("history", help="Show entity change history")
    p_history.add_argument("entity_id")
    p_history.add_argument("--attr", default=None)
    p_history.set_defaults(func=cmd_history)

    p_flush = sub.add_parser("flush", help="Flush memory pipeline")
    p_flush.add_argument("level", nargs="?", default="all",
                         choices=["raw", "canonical", "projection", "snapshot", "index", "all"])
    p_flush.set_defaults(func=cmd_flush)

    p_merge = sub.add_parser("merge", help="Merge duplicate entities")
    p_merge.add_argument("source_id")
    p_merge.add_argument("target_id")
    p_merge.add_argument("--reason", default=None)
    p_merge.set_defaults(func=cmd_merge)

    p_duplicates = sub.add_parser("duplicates", help="List duplicate entity candidates")
    p_duplicates.add_argument("--entity-id", default=None)
    p_duplicates.add_argument("--status", default="OPEN")
    p_duplicates.add_argument("--limit", type=int, default=100)
    p_duplicates.set_defaults(func=cmd_duplicates)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
