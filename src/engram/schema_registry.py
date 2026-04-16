from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import unicodedata
from typing import Any


_PROJECT_STATUS_MAP = {
    "planned": "planned",
    "plan": "planned",
    "예정": "planned",
    "계획": "planned",
    "in_progress": "in_progress",
    "in progress": "in_progress",
    "진행중": "in_progress",
    "진행 중": "in_progress",
    "working": "in_progress",
    "review": "review",
    "리뷰": "review",
    "검토": "review",
    "blocked": "blocked",
    "보류": "blocked",
    "막힘": "blocked",
    "done": "done",
    "complete": "done",
    "completed": "done",
    "finished": "done",
    "완료": "done",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "취소": "cancelled",
}

_KNOWN_TOOL_NAMES = {
    "figma": "Figma",
    "sketch": "Sketch",
    "photoshop": "Photoshop",
    "illustrator": "Illustrator",
}


def normalize_alias(value: str) -> str:
    compact = " ".join(value.strip().split()).casefold()
    compact = unicodedata.normalize("NFKC", compact)
    return compact


def slugify_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    if slug:
        return slug
    fallback = re.sub(r"\s+", "-", normalize_alias(value)).strip("-")
    if fallback:
        return fallback
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return digest


def _looks_like_project_name(value: str) -> bool:
    lowered = normalize_alias(value)
    return lowered.startswith("project ") or lowered.endswith(" 프로젝트") or lowered.endswith("프로젝트")


def _project_slug_from_name(value: str) -> str:
    lowered = normalize_alias(value)
    if lowered.startswith("project "):
        value = value.split(" ", 1)[1]
    if value.endswith(" 프로젝트"):
        value = value[: -len(" 프로젝트")]
    elif value.endswith("프로젝트"):
        value = value[: -len("프로젝트")]
    return slugify_ascii(value)


@dataclass(frozen=True, slots=True)
class EntitySchema:
    type_name: str
    id_prefix: str
    scalar_attrs: tuple[str, ...]
    list_attrs: tuple[str, ...] = ()
    alias_attr_names: tuple[str, ...] = ("name",)


@dataclass(frozen=True, slots=True)
class RelationSchema:
    relation_type: str
    allowed_attr_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizedEntityPayload:
    entity_id: str
    entity_type: str
    attrs: dict[str, Any]
    aliases: list[str]
    canonical_keys: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NormalizedRelationPayload:
    source: str
    target: str
    relation_type: str
    attrs: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SchemaRegistry:
    entity_schemas: dict[str, EntitySchema]
    relation_schemas: dict[str, RelationSchema]

    @classmethod
    def default(cls) -> "SchemaRegistry":
        entity_schemas = {
            "person": EntitySchema(
                type_name="person",
                id_prefix="person",
                scalar_attrs=("name", "origin", "residence"),
                list_attrs=("aliases", "tools"),
                alias_attr_names=("name",),
            ),
            "project": EntitySchema(
                type_name="project",
                id_prefix="project",
                scalar_attrs=("name", "status", "description"),
                alias_attr_names=("name",),
            ),
            "tool": EntitySchema(
                type_name="tool",
                id_prefix="tool",
                scalar_attrs=("name",),
                alias_attr_names=("name",),
            ),
            "location": EntitySchema(
                type_name="location",
                id_prefix="location",
                scalar_attrs=("name", "city", "country"),
                alias_attr_names=("name", "city"),
            ),
            "user": EntitySchema(
                type_name="user",
                id_prefix="user",
                scalar_attrs=("name", "origin", "residence"),
                list_attrs=("aliases", "tools"),
                alias_attr_names=("name",),
            ),
            "entity": EntitySchema(
                type_name="entity",
                id_prefix="entity",
                scalar_attrs=("name", "description"),
                list_attrs=("aliases",),
                alias_attr_names=("name",),
            ),
        }
        relation_schemas = {
            "works_on": RelationSchema("works_on", allowed_attr_names=("role", "responsibility", "level", "status")),
            "uses": RelationSchema("uses", allowed_attr_names=("role", "responsibility", "level", "status")),
            "assigned_to": RelationSchema("assigned_to", allowed_attr_names=("role", "responsibility", "level", "status")),
            "member": RelationSchema("member", allowed_attr_names=("role", "responsibility", "level", "status")),
            "manager": RelationSchema("manager", allowed_attr_names=("scope", "role")),
        }
        return cls(entity_schemas=entity_schemas, relation_schemas=relation_schemas)

    def summarize_for_extractor(self) -> dict[str, Any]:
        return {
            "entity_types": {
                key: {
                    "id_prefix": schema.id_prefix,
                    "scalar_attrs": list(schema.scalar_attrs),
                    "list_attrs": list(schema.list_attrs),
                }
                for key, schema in sorted(self.entity_schemas.items())
                if key not in {"entity", "user"}
            },
            "relation_types": {
                key: list(schema.allowed_attr_names)
                for key, schema in sorted(self.relation_schemas.items())
            },
            "project_status_values": [
                "planned",
                "in_progress",
                "review",
                "blocked",
                "done",
                "cancelled",
            ],
        }

    def normalize_entity(
        self,
        *,
        raw_id: str,
        raw_type: str | None,
        attrs: dict[str, Any],
        prefer_explicit_id: bool = True,
    ) -> NormalizedEntityPayload:
        attrs = dict(attrs)
        warnings: list[str] = []
        inferred_type = self.infer_entity_type(raw_type=raw_type, raw_id=raw_id, attrs=attrs)
        schema = self.entity_schemas.get(inferred_type, self.entity_schemas["entity"])
        normalized_attrs = self._normalize_entity_attrs(inferred_type, attrs, warnings)
        explicit_id = raw_id if ":" in raw_id else None
        if explicit_id and (prefer_explicit_id or inferred_type != "entity"):
            entity_id = self._canonicalize_explicit_id(explicit_id, inferred_type)
        else:
            entity_id = self._generate_entity_id(inferred_type, raw_id=raw_id, attrs=normalized_attrs)
        aliases = self._entity_aliases(schema, raw_id=raw_id, attrs=normalized_attrs)
        canonical_keys = self._entity_canonical_keys(inferred_type, entity_id=entity_id, attrs=normalized_attrs)
        return NormalizedEntityPayload(
            entity_id=entity_id,
            entity_type=inferred_type,
            attrs=normalized_attrs,
            aliases=aliases,
            canonical_keys=canonical_keys,
            warnings=warnings,
        )

    def normalize_relation(
        self,
        *,
        source: str,
        target: str,
        relation_type: str,
        attrs: dict[str, Any],
    ) -> NormalizedRelationPayload:
        schema = self.relation_schemas.get(relation_type)
        warnings: list[str] = []
        normalized_attrs = dict(attrs)
        if schema is not None and normalized_attrs:
            unknown_keys = sorted(set(normalized_attrs) - set(schema.allowed_attr_names))
            if unknown_keys:
                warnings.append(
                    f"unrecognized relation attrs preserved for {relation_type}: {', '.join(unknown_keys)}"
                )
        return NormalizedRelationPayload(
            source=source,
            target=target,
            relation_type=relation_type,
            attrs=normalized_attrs,
            warnings=warnings,
        )

    def infer_entity_type(self, *, raw_type: str | None, raw_id: str, attrs: dict[str, Any]) -> str:
        explicit = (raw_type or "").strip().lower()
        if explicit and explicit != "entity":
            return explicit
        if ":" in raw_id:
            prefix = raw_id.split(":", 1)[0].strip().lower()
            if prefix and prefix != "entity":
                return prefix

        raw_name = str(attrs.get("name") or raw_id or "")
        lowered_name = normalize_alias(raw_name)
        if _looks_like_project_name(raw_name) or "status" in attrs:
            return "project"
        if lowered_name in _KNOWN_TOOL_NAMES or raw_name.lower() in _KNOWN_TOOL_NAMES:
            return "tool"
        if any(key in attrs for key in ("origin", "residence", "location")):
            return "person"
        if explicit == "user":
            return "user"
        return "person" if explicit == "person" else "entity"

    def normalize_attr_value(self, key: str, value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.strip().split())
        if isinstance(value, list):
            normalized = [self.normalize_attr_value(key, item) for item in value]
            deduped: list[Any] = []
            seen: set[str] = set()
            for item in normalized:
                marker = json_marker(item)
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(item)
            return deduped
        return value

    def _normalize_entity_attrs(self, entity_type: str, attrs: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, raw_value in attrs.items():
            value = self.normalize_attr_value(key, raw_value)
            if entity_type in {"person", "user"} and key == "residence":
                normalized["location"] = value
                warnings.append("mapped residence to location")
                continue
            if entity_type in {"person", "user"} and key == "tool":
                normalized.setdefault("tools", [])
                if isinstance(value, list):
                    normalized["tools"] = _union_lists(normalized["tools"], value)
                else:
                    normalized["tools"] = _union_lists(normalized["tools"], [value])
                warnings.append("mapped tool to tools[]")
                continue
            if entity_type == "project" and key == "status" and isinstance(value, str):
                normalized["status"] = _PROJECT_STATUS_MAP.get(normalize_alias(value), normalize_alias(value).replace(" ", "_"))
                continue
            if entity_type == "tool" and key == "name" and isinstance(value, str):
                normalized["name"] = _KNOWN_TOOL_NAMES.get(normalize_alias(value), value)
                continue
            normalized[key] = value
        return normalized

    def _generate_entity_id(self, entity_type: str, *, raw_id: str, attrs: dict[str, Any]) -> str:
        if entity_type == "project":
            name = str(attrs.get("name") or raw_id)
            return f"project:{_project_slug_from_name(name)}"
        if entity_type == "tool":
            name = str(attrs.get("name") or raw_id)
            return f"tool:{slugify_ascii(name)}"
        if entity_type == "location":
            name = str(attrs.get("name") or attrs.get("city") or raw_id)
            return f"location:{slugify_ascii(name)}"
        if entity_type in {"person", "user"}:
            name = str(attrs.get("name") or raw_id)
            return f"{self.entity_schemas[entity_type].id_prefix}:{slugify_ascii(name)}"
        if entity_type != "entity":
            name = str(attrs.get("name") or raw_id)
            return f"{entity_type}:{slugify_ascii(name)}"
        name = str(attrs.get("name") or raw_id)
        return f"entity:{slugify_ascii(name)}"

    def _canonicalize_explicit_id(self, explicit_id: str, entity_type: str) -> str:
        prefix, suffix = explicit_id.split(":", 1)
        if entity_type == "project" and suffix.startswith("project-"):
            suffix = suffix[len("project-"):]
        if entity_type == "tool" and suffix.startswith("tool-"):
            suffix = suffix[len("tool-"):]
        if entity_type == "location" and suffix.startswith("location-"):
            suffix = suffix[len("location-"):]
        target_prefix = self.entity_schemas.get(entity_type, EntitySchema(entity_type, entity_type, ("name",))).id_prefix
        if prefix in {"entity", "person", "project", "tool", "location", "user", entity_type}:
            if prefix != target_prefix and prefix == "entity":
                return f"{target_prefix}:{slugify_ascii(suffix)}"
            if prefix == target_prefix:
                return f"{target_prefix}:{slugify_ascii(suffix)}"
            return explicit_id
        return f"{target_prefix}:{slugify_ascii(suffix)}"

    def _entity_aliases(self, schema: EntitySchema, *, raw_id: str, attrs: dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        if raw_id and ":" not in raw_id:
            aliases.append(raw_id)
        if ":" in raw_id:
            aliases.append(raw_id.split(":", 1)[1])
        for name in schema.alias_attr_names:
            raw = attrs.get(name)
            if isinstance(raw, str) and raw.strip():
                aliases.append(raw)
        for alias in attrs.get("aliases", []) if isinstance(attrs.get("aliases"), list) else []:
            if isinstance(alias, str) and alias.strip():
                aliases.append(alias)
        return _dedupe_strings(aliases)

    def _entity_canonical_keys(self, entity_type: str, *, entity_id: str, attrs: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        keys.append(entity_id.split(":", 1)[1])
        name = attrs.get("name")
        if entity_type == "project" and isinstance(name, str):
            keys.append(_project_slug_from_name(name))
        elif isinstance(name, str) and name.strip():
            keys.append(slugify_ascii(name))
        return _dedupe_strings(keys)


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        marker = normalize_alias(value)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped


def _union_lists(existing: list[Any], values: list[Any]) -> list[Any]:
    merged = list(existing)
    seen = {json_marker(item) for item in merged}
    for value in values:
        marker = json_marker(value)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(value)
    return merged


def json_marker(value: Any) -> str:
    return repr(value)
