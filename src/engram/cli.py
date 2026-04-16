from __future__ import annotations

import argparse
import json
import os
import sys

from .canonical import NullExtractor
from .engram import Engram
from .meaning_index import NullMeaningAnalyzer
from .semantic import HashEmbedder
from .time_utils import from_rfc3339, to_rfc3339


def _build_engram(args) -> Engram:
    return Engram(
        user_id=args.user_id,
        path=args.path,
        extractor=_build_extractor(),
        embedder=_build_embedder(),
        meaning_analyzer=_build_meaning_analyzer(),
        auto_flush=False,
    )


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
