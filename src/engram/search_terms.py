from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .types import Event

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣:_-]+")
KOREAN_SUFFIXES = (
    "으로는",
    "으로도",
    "으로",
    "에서",
    "에게",
    "한테",
    "이랑",
    "처럼",
    "까지",
    "부터",
    "보다",
    "에는",
    "에도",
    "으로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "로",
    "도",
    "만",
    "야",
)


@dataclass(frozen=True, slots=True)
class QueryToken:
    raw: str
    variants: tuple[str, ...]


def event_search_text(event: Event) -> str:
    parts = [
        event.type,
        json.dumps(event.data, ensure_ascii=False, sort_keys=True),
        event.reason or "",
        event.source_role,
    ]
    return " ".join(parts).lower()


def query_tokens(query: str) -> list[QueryToken]:
    tokens: list[QueryToken] = []
    for raw_token in TOKEN_RE.findall(query):
        token = raw_token.lower().strip()
        if not token:
            continue
        tokens.append(QueryToken(raw=token, variants=_token_variants(token)))
    return tokens


def query_candidate_terms(query: str) -> list[str]:
    candidates: list[str] = []
    for token in query_tokens(query):
        candidates.extend(token.variants)
        for variant in token.variants:
            candidates.extend(_split_token_parts(variant))
    return list(dict.fromkeys(candidates))


def event_search_terms(event: Event) -> list[str]:
    return search_terms_from_text(event_search_text(event))


def search_terms_from_text(text: str) -> list[str]:
    terms: list[str] = []
    for raw_token in TOKEN_RE.findall(text):
        token = raw_token.lower().strip()
        if not token:
            continue
        terms.extend(_token_variants(token))
        terms.extend(_split_token_parts(token))
    return list(dict.fromkeys(term for term in terms if term))


def _token_variants(token: str) -> tuple[str, ...]:
    variants = [token]
    if _contains_hangul(token):
        for suffix in KOREAN_SUFFIXES:
            if token.endswith(suffix):
                stem = token[: -len(suffix)]
                if len(stem) >= 2:
                    variants.append(stem)
    return tuple(dict.fromkeys(variants))


def _split_token_parts(token: str) -> list[str]:
    parts = [part for part in re.split(r"[:_-]+", token) if len(part) >= 2]
    return list(dict.fromkeys(parts))


def _contains_hangul(value: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in value)
