from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import ValidationError
from .event_ops import validate_event
from .semantic import _default_openai_space_id, _load_openai_client_class
from .types import ExtractedEvent, QueueItem, RawTurn

_OPENAI_DEFAULT_MODEL = "gpt-5.4-mini"
_SELF_MARKERS = {
    "self",
    "me",
    "myself",
    "current_user",
    "current-user",
    "나",
    "저",
    "본인",
}


@dataclass(slots=True)
class OpenAIExtractor:
    api_key: str | None = None
    model: str = _OPENAI_DEFAULT_MODEL
    base_url: str | None = None
    temperature: float = 0.0
    version: str = field(init=False)
    _client: Any = field(init=False, default=None, repr=False)
    _space_id: str = field(init=False, repr=False)
    _safe_user_id: str | None = field(init=False, default=None, repr=False)
    _recent_turns_provider: Callable[[QueueItem, int], list[RawTurn]] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._space_id = _default_openai_space_id(self.base_url)
        self.version = f"openai-extractor:{self._space_id}:{self.model}:v1"
        self._client = None
        self._safe_user_id: str | None = None
        self._recent_turns_provider: Callable[[QueueItem, int], list[RawTurn]] | None = None

    def bind_runtime_context(
        self,
        *,
        safe_user_id: str,
        recent_turns_provider: Callable[[QueueItem, int], list[RawTurn]],
    ) -> None:
        self._safe_user_id = safe_user_id
        self._recent_turns_provider = recent_turns_provider

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        if self._safe_user_id is None or self._recent_turns_provider is None:
            raise RuntimeError("OpenAIExtractor must be bound to Engram runtime context before use")

        prompt = self._build_prompt(
            item=item,
            recent_turns=self._recent_turns_provider(item, 4),
            safe_user_id=self._safe_user_id,
        )
        response = self._client_instance().chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
        )
        content = _response_text(response)
        extracted = _parse_extracted_events(content, safe_user_id=self._safe_user_id)
        return _normalize_event_batch(extracted)

    def _build_prompt(
        self,
        *,
        item: QueueItem,
        recent_turns: list[RawTurn],
        safe_user_id: str,
    ) -> str:
        payload = {
            "current_user_entity_id": f"user:{safe_user_id}",
            "rules": {
                "only_extract_explicit_facts": True,
                "no_guessing_or_world_knowledge": True,
                "if_ambiguous_return_no_events": True,
                "no_causal_output": True,
                "no_effective_time_output": True,
                "source_role": "user",
            },
            "id_policy": {
                "self_reference_marker": "self",
                "self_entity_id": f"user:{safe_user_id}",
                "other_people": "person:<slug>",
                "non_people_entities": "entity:<slug>",
            },
            "allowed_event_types": {
                "entity.create": {"data": {"id": "string", "type": "string", "attrs": "object"}},
                "entity.update": {"data": {"id": "string", "attrs": "object"}},
                "entity.delete": {"data": {"id": "string"}},
                "relation.create": {
                    "data": {"source": "string", "target": "string", "type": "string", "attrs": "object"}
                },
                "relation.update": {
                    "data": {"source": "string", "target": "string", "type": "string", "attrs": "object"}
                },
                "relation.delete": {"data": {"source": "string", "target": "string", "type": "string"}},
            },
            "required_output_shape": {
                "events": [
                    {
                        "type": "allowed event type",
                        "data": "event payload object",
                        "confidence": "float 0.0..1.0",
                        "reason": "short natural-language reason",
                    }
                ]
            },
            "turn": {
                "turn_id": item.turn_id,
                "session_id": item.session_id,
                "observed_at": item.observed_at.isoformat().replace("+00:00", "Z"),
                "user": item.user,
                "assistant": item.assistant,
                "metadata": item.metadata,
            },
            "recent_turns": [
                {
                    "turn_id": turn.id,
                    "session_id": turn.session_id,
                    "observed_at": turn.observed_at.isoformat().replace("+00:00", "Z"),
                    "user": turn.user,
                    "assistant": turn.assistant,
                }
                for turn in recent_turns
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    def _client_instance(self):
        if self._client is None:
            client_class = _load_openai_client_class()
            self._client = client_class(api_key=self.api_key, base_url=self.base_url)
        return self._client


def _system_prompt() -> str:
    return (
        "You extract structured memory events from one chat turn.\n"
        "Return JSON only.\n"
        "Extract only facts explicitly stated in the provided turn.\n"
        "Do not infer, imagine, or fill gaps with common sense.\n"
        "If the statement is ambiguous, return {\"events\": []}.\n"
        "Do not output caused_by.\n"
        "Do not output effective_at_start or effective_at_end.\n"
        "Use 'self' for the current user when needed.\n"
        "Prefer conservative updates over aggressive rewriting.\n"
    )


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValidationError("OpenAIExtractor response did not contain any choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    raise ValidationError("OpenAIExtractor response did not contain textual JSON content")


def _parse_extracted_events(content: str, *, safe_user_id: str) -> list[ExtractedEvent]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"OpenAIExtractor returned malformed JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValidationError("OpenAIExtractor JSON must be an object with an 'events' array")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValidationError("OpenAIExtractor JSON must contain an 'events' array")

    parsed: list[ExtractedEvent] = []
    for index, raw_event in enumerate(raw_events):
        parsed.append(_parse_event(raw_event, safe_user_id=safe_user_id, index=index))
    return parsed


def _parse_event(raw_event: Any, *, safe_user_id: str, index: int) -> ExtractedEvent:
    if not isinstance(raw_event, dict):
        raise ValidationError(f"event[{index}] must be an object")
    allowed_keys = {"type", "data", "confidence", "reason"}
    unknown_keys = set(raw_event) - allowed_keys
    if unknown_keys:
        raise ValidationError(f"event[{index}] contains unsupported keys: {sorted(unknown_keys)}")

    event_type = raw_event.get("type")
    data = raw_event.get("data")
    confidence = raw_event.get("confidence")
    reason = raw_event.get("reason")

    if not isinstance(event_type, str) or not event_type.strip():
        raise ValidationError(f"event[{index}].type must be a non-empty string")
    if not isinstance(data, dict):
        raise ValidationError(f"event[{index}].data must be an object")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValidationError(f"event[{index}].confidence must be a number between 0.0 and 1.0")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValidationError(f"event[{index}].confidence must be between 0.0 and 1.0")
    if not isinstance(reason, str) or not reason.strip():
        raise ValidationError(f"event[{index}].reason must be a non-empty string")

    normalized_data = _normalize_event_data(event_type.strip(), data, safe_user_id=safe_user_id)
    validate_event(event_type.strip(), normalized_data)
    return ExtractedEvent(
        type=event_type.strip(),
        data=normalized_data,
        effective_at_start=None,
        effective_at_end=None,
        caused_by=None,
        source_role="user",
        confidence=float(confidence),
        reason=reason.strip(),
        time_confidence="unknown",
    )


def _normalize_event_data(event_type: str, data: dict[str, Any], *, safe_user_id: str) -> dict[str, Any]:
    if event_type == "entity.create":
        raw_id = _require_string(data, "id")
        entity_type = _require_string(data, "type").strip().lower()
        attrs = _require_object(data, "attrs")
        normalized_id, is_self = _normalize_entity_id(raw_id, entity_type=entity_type, safe_user_id=safe_user_id)
        normalized_type = "user" if is_self else entity_type
        return {"id": normalized_id, "type": normalized_type, "attrs": attrs}

    if event_type == "entity.update":
        raw_id = _require_string(data, "id")
        attrs = _require_object(data, "attrs")
        normalized_id, _ = _normalize_entity_id(raw_id, entity_type=None, safe_user_id=safe_user_id)
        return {"id": normalized_id, "attrs": attrs}

    if event_type == "entity.delete":
        raw_id = _require_string(data, "id")
        normalized_id, _ = _normalize_entity_id(raw_id, entity_type=None, safe_user_id=safe_user_id)
        return {"id": normalized_id}

    if event_type in {"relation.create", "relation.update"}:
        source = _normalize_relation_endpoint(_require_string(data, "source"), safe_user_id=safe_user_id)
        target = _normalize_relation_endpoint(_require_string(data, "target"), safe_user_id=safe_user_id)
        relation_type = _require_string(data, "type").strip()
        attrs = _require_object(data, "attrs")
        return {"source": source, "target": target, "type": relation_type, "attrs": attrs}

    if event_type == "relation.delete":
        source = _normalize_relation_endpoint(_require_string(data, "source"), safe_user_id=safe_user_id)
        target = _normalize_relation_endpoint(_require_string(data, "target"), safe_user_id=safe_user_id)
        relation_type = _require_string(data, "type").strip()
        return {"source": source, "target": target, "type": relation_type}

    return dict(data)


def _normalize_event_batch(events: list[ExtractedEvent]) -> list[ExtractedEvent]:
    normalized: list[ExtractedEvent] = []
    entity_update_indexes: dict[str, int] = {}
    relation_update_indexes: dict[tuple[str, str, str], int] = {}

    for event in events:
        if event.type == "entity.update":
            attrs = dict(event.data["attrs"])
            if not attrs:
                continue
            entity_id = event.data["id"]
            index = entity_update_indexes.get(entity_id)
            if index is not None:
                _merge_update_event(normalized[index], event)
                continue
            entity_update_indexes[entity_id] = len(normalized)
            normalized.append(event)
            continue

        if event.type == "relation.update":
            attrs = dict(event.data["attrs"])
            if not attrs:
                continue
            key = (event.data["source"], event.data["target"], event.data["type"])
            index = relation_update_indexes.get(key)
            if index is not None:
                _merge_update_event(normalized[index], event)
                continue
            relation_update_indexes[key] = len(normalized)
            normalized.append(event)
            continue

        if event.type.startswith("entity."):
            entity_update_indexes.pop(event.data["id"], None)
        if event.type.startswith("relation."):
            key = (event.data["source"], event.data["target"], event.data["type"])
            relation_update_indexes.pop(key, None)
        normalized.append(event)
    return normalized


def _merge_update_event(target: ExtractedEvent, incoming: ExtractedEvent) -> None:
    target.data["attrs"].update(incoming.data["attrs"])
    if target.confidence is None:
        target.confidence = incoming.confidence
    elif incoming.confidence is not None:
        target.confidence = max(target.confidence, incoming.confidence)
    if incoming.reason:
        target.reason = incoming.reason


def _normalize_entity_id(
    raw_id: str,
    *,
    entity_type: str | None,
    safe_user_id: str,
) -> tuple[str, bool]:
    candidate = raw_id.strip()
    if not candidate:
        raise ValidationError("entity id must not be empty")
    if _is_self_reference(candidate, safe_user_id=safe_user_id):
        return f"user:{safe_user_id}", True
    if ":" in candidate:
        return candidate, False

    slug = _slugify(candidate)
    if not slug:
        raise ValidationError("entity id must not normalize to empty")
    if entity_type in {"user", "person"}:
        return f"person:{slug}", False
    return f"entity:{slug}", False


def _normalize_relation_endpoint(raw_id: str, *, safe_user_id: str) -> str:
    candidate = raw_id.strip()
    if not candidate:
        raise ValidationError("relation endpoint must not be empty")
    if _is_self_reference(candidate, safe_user_id=safe_user_id):
        return f"user:{safe_user_id}"
    if ":" in candidate:
        return candidate
    slug = _slugify(candidate)
    if not slug:
        raise ValidationError("relation endpoint must not normalize to empty")
    return f"person:{slug}"


def _is_self_reference(value: str, *, safe_user_id: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return normalized in _SELF_MARKERS or normalized in {
        safe_user_id.lower(),
        f"user:{safe_user_id}".lower(),
    }


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    collapsed = re.sub(r"\s+", "-", normalized)
    cleaned = re.sub(r"[^\w가-힣.-]", "-", collapsed, flags=re.UNICODE)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-._")


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{key} must be a non-empty string")
    return value


def _require_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValidationError(f"{key} must be an object")
    return dict(value)
