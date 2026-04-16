from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from .schema_registry import NormalizedEntityPayload, SchemaRegistry, normalize_alias
from .time_utils import utcnow
from .types import DuplicateCandidate, Event, ExtractedEvent


@dataclass(frozen=True, slots=True)
class ResolvedCandidate:
    canonical_entity_id: str
    canonical_entity_type: str
    matched_by: str | None
    matched_existing: bool
    aliases_to_add: tuple[str, ...] = ()
    canonical_keys_to_add: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    duplicate_target_id: str | None = None


@dataclass(slots=True)
class ResolvedBatch:
    drafts: list[ExtractedEvent] = field(default_factory=list)
    aliases: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidate] = field(default_factory=list)


class DataQualityManager:
    def __init__(self, store, schema_registry: SchemaRegistry):
        self.store = store
        self.schema_registry = schema_registry

    def process_drafts(
        self,
        drafts: list[ExtractedEvent],
        *,
        observed_at: datetime,
        source_turn_id: str | None,
        keep_noop_updates: bool = False,
    ) -> ResolvedBatch:
        batch = ResolvedBatch()
        batch_entity_map: dict[str, ResolvedCandidate] = {}
        kept_indexes: list[int] = []
        for index, draft in enumerate(drafts):
            resolved = self._resolve_draft(
                draft,
                observed_at=observed_at,
                source_turn_id=source_turn_id,
                batch_entity_map=batch_entity_map,
                keep_noop_updates=keep_noop_updates,
            )
            batch.aliases.extend(resolved.aliases)
            batch.duplicate_candidates.extend(resolved.duplicate_candidates)
            if resolved.draft is None:
                continue
            kept_indexes.append(index)
            batch.drafts.append(resolved.draft)
            if resolved.batch_entities:
                batch_entity_map.update(resolved.batch_entities)

        if kept_indexes == list(range(len(drafts))):
            return batch

        old_to_new = {old_index: new_index for new_index, old_index in enumerate(kept_indexes)}
        remapped: list[ExtractedEvent] = []
        for draft in batch.drafts:
            caused_by = draft.caused_by
            if caused_by is None or not caused_by.startswith("__batch_ref:"):
                remapped.append(draft)
                continue
            old_index = int(caused_by.split(":", 1)[1])
            new_index = old_to_new.get(old_index)
            remapped.append(
                ExtractedEvent(
                    type=draft.type,
                    data=dict(draft.data),
                    effective_at_start=draft.effective_at_start,
                    effective_at_end=draft.effective_at_end,
                    caused_by=f"__batch_ref:{new_index}" if new_index is not None else None,
                    source_role=draft.source_role,
                    confidence=draft.confidence,
                    reason=draft.reason,
                    time_confidence=draft.time_confidence,
                )
            )
        batch.drafts = remapped
        return batch

    def normalize_manual_event(
        self,
        *,
        event_type: str,
        data: dict[str, Any],
        observed_at: datetime,
    ) -> ResolvedBatch:
        batch = self.process_drafts(
            [ExtractedEvent(type=event_type, data=data, source_role="manual")],
            observed_at=observed_at,
            source_turn_id=None,
            keep_noop_updates=True,
        )
        return batch

    def merge_entities(self, source_id: str, target_id: str, *, reason: str | None = None) -> str:
        now = utcnow()
        canonical_target = self.store.resolve_redirect_target(target_id)
        canonical_source = self.store.resolve_redirect_target(source_id)
        if canonical_target == canonical_source:
            return canonical_target
        source_aliases = self.store.list_alias_rows_for_entity(canonical_source)
        with self.store.transaction() as tx:
            self.store.add_entity_redirect(
                tx,
                source_entity_id=canonical_source,
                target_entity_id=canonical_target,
                merged_at=now,
                reason=reason,
            )
            if source_aliases:
                self.store.append_entity_alias_rows(
                    tx,
                    [
                        (
                            canonical_target,
                            row.entity_type,
                            row.alias,
                            row.normalized_alias,
                            row.alias_kind,
                            now,
                        )
                        for row in source_aliases
                    ],
                )
            self.store.resolve_duplicate_candidates_for_merge(
                tx,
                source_entity_id=canonical_source,
                target_entity_id=canonical_target,
            )
        return canonical_target

    def _resolve_draft(
        self,
        draft: ExtractedEvent,
        *,
        observed_at: datetime,
        source_turn_id: str | None,
        batch_entity_map: dict[str, ResolvedCandidate],
        keep_noop_updates: bool,
    ) -> "_ResolvedDraft":
        if draft.type.startswith("entity."):
            return self._resolve_entity_draft(
                draft,
                observed_at=observed_at,
                source_turn_id=source_turn_id,
                batch_entity_map=batch_entity_map,
                keep_noop_updates=keep_noop_updates,
            )
        if draft.type.startswith("relation."):
            return self._resolve_relation_draft(
                draft,
                observed_at=observed_at,
                source_turn_id=source_turn_id,
                batch_entity_map=batch_entity_map,
            )
        return _ResolvedDraft(draft=draft)

    def _resolve_entity_draft(
        self,
        draft: ExtractedEvent,
        *,
        observed_at: datetime,
        source_turn_id: str | None,
        batch_entity_map: dict[str, ResolvedCandidate],
        keep_noop_updates: bool,
    ) -> "_ResolvedDraft":
        raw_id = str(draft.data["id"])
        raw_type = str(draft.data.get("type")) if draft.type == "entity.create" and draft.data.get("type") else None
        normalized = self.schema_registry.normalize_entity(
            raw_id=raw_id,
            raw_type=raw_type,
            attrs=dict(draft.data.get("attrs", {})),
            prefer_explicit_id=draft.type == "entity.create",
        )
        resolved = self._resolve_entity_candidate(
            normalized=normalized,
            batch_entity_map=batch_entity_map,
        )
        if resolved.duplicate_target_id is not None and draft.type == "entity.update":
            return _ResolvedDraft(
                draft=None,
                aliases=self._alias_rows(resolved, normalized),
                duplicate_candidates=self._duplicate_rows(
                    entity_id=resolved.canonical_entity_id,
                    candidate_entity_id=resolved.duplicate_target_id,
                    observed_at=observed_at,
                    source_turn_id=source_turn_id,
                    event_type=draft.type,
                    reason="ambiguous entity update candidate",
                    score=0.6,
                    match_basis="ambiguous_alias",
                ),
            )

        if draft.type == "entity.delete":
            delete_id = resolved.canonical_entity_id
            if resolved.duplicate_target_id is not None:
                return _ResolvedDraft(
                    draft=None,
                    aliases=self._alias_rows(resolved, normalized),
                    duplicate_candidates=self._duplicate_rows(
                        entity_id=resolved.canonical_entity_id,
                        candidate_entity_id=resolved.duplicate_target_id,
                        observed_at=observed_at,
                        source_turn_id=source_turn_id,
                        event_type=draft.type,
                        reason="ambiguous entity delete candidate",
                        score=0.6,
                        match_basis="ambiguous_alias",
                    ),
                    batch_entities={raw_id: resolved},
                )
            return _ResolvedDraft(
                draft=ExtractedEvent(
                    type="entity.delete",
                    data={"id": delete_id},
                    effective_at_start=draft.effective_at_start,
                    effective_at_end=draft.effective_at_end,
                    caused_by=draft.caused_by,
                    source_role=draft.source_role,
                    confidence=draft.confidence,
                    reason=draft.reason,
                    time_confidence=draft.time_confidence,
                ),
                aliases=self._alias_rows(resolved, normalized),
                duplicate_candidates=self._duplicate_rows_from_resolved(
                    resolved,
                    observed_at=observed_at,
                    source_turn_id=source_turn_id,
                    event_type=draft.type,
                ),
                batch_entities={raw_id: resolved},
            )

        attrs = dict(normalized.attrs)
        preserve_explicit_create = (
            draft.type == "entity.create"
            and resolved.matched_existing
            and resolved.matched_by == "explicit_id"
            and source_turn_id is not None
        )
        if resolved.matched_existing:
            if preserve_explicit_create:
                draft = ExtractedEvent(
                    type="entity.create",
                    data={
                        "id": resolved.canonical_entity_id,
                        "type": resolved.canonical_entity_type,
                        "attrs": attrs,
                    },
                    effective_at_start=draft.effective_at_start,
                    effective_at_end=draft.effective_at_end,
                    caused_by=draft.caused_by,
                    source_role=draft.source_role,
                    confidence=draft.confidence,
                    reason=draft.reason,
                    time_confidence=draft.time_confidence,
                )
            else:
                current = self.store.materialize_current_entity(resolved.canonical_entity_id)
                merged_attrs = self._merge_attrs(
                    entity_type=resolved.canonical_entity_type,
                    current_attrs=current.attrs if current is not None else {},
                    new_attrs=attrs,
                )
                changed_attrs = {
                    key: value
                    for key, value in merged_attrs.items()
                    if (current is None or current.attrs.get(key) != value)
                }
                if not changed_attrs and not keep_noop_updates:
                    return _ResolvedDraft(
                        draft=None,
                        aliases=self._alias_rows(resolved, normalized),
                        duplicate_candidates=self._duplicate_rows_from_resolved(
                            resolved,
                            observed_at=observed_at,
                            source_turn_id=source_turn_id,
                            event_type=draft.type,
                        ),
                        batch_entities={raw_id: resolved},
                    )
                if not changed_attrs:
                    changed_attrs = attrs
                draft = ExtractedEvent(
                    type="entity.update",
                    data={"id": resolved.canonical_entity_id, "attrs": changed_attrs},
                    effective_at_start=draft.effective_at_start,
                    effective_at_end=draft.effective_at_end,
                    caused_by=draft.caused_by,
                    source_role=draft.source_role,
                    confidence=draft.confidence,
                    reason=draft.reason,
                    time_confidence=draft.time_confidence,
                )
        else:
            create_type = resolved.canonical_entity_type
            draft = ExtractedEvent(
                type="entity.create",
                data={"id": resolved.canonical_entity_id, "type": create_type, "attrs": attrs},
                effective_at_start=draft.effective_at_start,
                effective_at_end=draft.effective_at_end,
                caused_by=draft.caused_by,
                source_role=draft.source_role,
                confidence=draft.confidence,
                reason=draft.reason,
                time_confidence=draft.time_confidence,
            )
        return _ResolvedDraft(
            draft=draft,
            aliases=self._alias_rows(resolved, normalized),
            duplicate_candidates=self._duplicate_rows_from_resolved(
                resolved,
                observed_at=observed_at,
                source_turn_id=source_turn_id,
                event_type=draft.type,
            ),
            batch_entities={raw_id: resolved},
        )

    def _resolve_relation_draft(
        self,
        draft: ExtractedEvent,
        *,
        observed_at: datetime,
        source_turn_id: str | None,
        batch_entity_map: dict[str, ResolvedCandidate],
    ) -> "_ResolvedDraft":
        source_raw = str(draft.data["source"])
        target_raw = str(draft.data["target"])
        source_id = self._resolve_relation_endpoint(source_raw, batch_entity_map)
        target_id = self._resolve_relation_endpoint(target_raw, batch_entity_map)
        relation = self.schema_registry.normalize_relation(
            source=source_id,
            target=target_id,
            relation_type=str(draft.data["type"]),
            attrs=dict(draft.data.get("attrs", {})),
        )
        draft = ExtractedEvent(
            type=draft.type,
            data={
                "source": relation.source,
                "target": relation.target,
                "type": relation.relation_type,
                "attrs": relation.attrs,
            },
            effective_at_start=draft.effective_at_start,
            effective_at_end=draft.effective_at_end,
            caused_by=draft.caused_by,
            source_role=draft.source_role,
            confidence=draft.confidence,
            reason=draft.reason,
            time_confidence=draft.time_confidence,
        )
        return _ResolvedDraft(draft=draft)

    def _resolve_relation_endpoint(
        self,
        raw_endpoint: str,
        batch_entity_map: dict[str, ResolvedCandidate],
    ) -> str:
        if raw_endpoint in batch_entity_map:
            return batch_entity_map[raw_endpoint].canonical_entity_id
        if ":" in raw_endpoint:
            canonical = self.store.resolve_redirect_target(raw_endpoint)
            return canonical
        normalized_alias = normalize_alias(raw_endpoint)
        matches = self.store.lookup_entities_by_alias(normalized_alias, entity_type=None, alias_kind=None)
        if len(matches) == 1:
            return matches[0]
        return raw_endpoint

    def _resolve_entity_candidate(
        self,
        *,
        normalized: NormalizedEntityPayload,
        batch_entity_map: dict[str, ResolvedCandidate],
    ) -> ResolvedCandidate:
        explicit_id = normalized.entity_id
        explicit_canonical = self.store.resolve_redirect_target(explicit_id)
        if self.store.entity_exists(explicit_canonical):
            return ResolvedCandidate(
                canonical_entity_id=explicit_canonical,
                canonical_entity_type=normalized.entity_type,
                matched_by="explicit_id",
                matched_existing=True,
                aliases_to_add=tuple(normalized.aliases),
                canonical_keys_to_add=tuple(normalized.canonical_keys),
                warnings=tuple(normalized.warnings),
            )

        for raw_id, candidate in batch_entity_map.items():
            if normalize_alias(raw_id) == normalize_alias(explicit_id):
                return candidate

        alias_matches: list[str] = []
        for alias in normalized.aliases:
            alias_matches.extend(
                self.store.lookup_entities_by_alias(
                    normalize_alias(alias),
                    entity_type=normalized.entity_type,
                    alias_kind=None,
                )
            )
        alias_matches = sorted(set(self.store.resolve_redirect_target(match) for match in alias_matches))
        if len(alias_matches) == 1:
            return ResolvedCandidate(
                canonical_entity_id=alias_matches[0],
                canonical_entity_type=normalized.entity_type,
                matched_by="alias",
                matched_existing=True,
                aliases_to_add=tuple(normalized.aliases),
                canonical_keys_to_add=tuple(normalized.canonical_keys),
                warnings=tuple(normalized.warnings),
            )

        canonical_key_matches: list[str] = []
        for key in normalized.canonical_keys:
            canonical_key_matches.extend(
                self.store.lookup_entities_by_alias(
                    normalize_alias(key),
                    entity_type=normalized.entity_type,
                    alias_kind="canonical_key",
                )
            )
        canonical_key_matches = sorted(
            set(self.store.resolve_redirect_target(match) for match in canonical_key_matches)
        )
        if len(canonical_key_matches) == 1:
            return ResolvedCandidate(
                canonical_entity_id=canonical_key_matches[0],
                canonical_entity_type=normalized.entity_type,
                matched_by="canonical_key",
                matched_existing=True,
                aliases_to_add=tuple(normalized.aliases),
                canonical_keys_to_add=tuple(normalized.canonical_keys),
                warnings=tuple(normalized.warnings),
            )

        duplicate_target = None
        if alias_matches:
            duplicate_target = alias_matches[0]
        elif canonical_key_matches:
            duplicate_target = canonical_key_matches[0]
        return ResolvedCandidate(
            canonical_entity_id=normalized.entity_id,
            canonical_entity_type=normalized.entity_type,
            matched_by=None,
            matched_existing=False,
            aliases_to_add=tuple(normalized.aliases),
            canonical_keys_to_add=tuple(normalized.canonical_keys),
            warnings=tuple(normalized.warnings),
            duplicate_target_id=duplicate_target,
        )

    def _merge_attrs(
        self,
        *,
        entity_type: str,
        current_attrs: dict[str, Any],
        new_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(current_attrs)
        list_attrs = set(self.schema_registry.entity_schemas.get(entity_type, self.schema_registry.entity_schemas["entity"]).list_attrs)
        for key, value in new_attrs.items():
            if key in list_attrs:
                existing = merged.get(key)
                if isinstance(existing, list) and isinstance(value, list):
                    merged[key] = _union_lists(existing, value)
                elif isinstance(existing, list):
                    merged[key] = _union_lists(existing, [value])
                elif isinstance(value, list):
                    merged[key] = _union_lists([], value if value is not None else [])
                else:
                    merged[key] = _union_lists([], [value])
                continue
            merged[key] = value
        return merged

    def _alias_rows(
        self,
        resolved: ResolvedCandidate,
        normalized: NormalizedEntityPayload,
    ) -> list[tuple[str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str]] = []
        for alias in resolved.aliases_to_add:
            rows.append(
                (
                    resolved.canonical_entity_id,
                    resolved.canonical_entity_type,
                    alias,
                    normalize_alias(alias),
                    "alias",
                )
            )
        for key in resolved.canonical_keys_to_add:
            rows.append(
                (
                    resolved.canonical_entity_id,
                    resolved.canonical_entity_type,
                    key,
                    normalize_alias(key),
                    "canonical_key",
                )
            )
        rows.append(
            (
                resolved.canonical_entity_id,
                resolved.canonical_entity_type,
                resolved.canonical_entity_id,
                normalize_alias(resolved.canonical_entity_id.split(":", 1)[1]),
                "id",
            )
        )
        name = normalized.attrs.get("name")
        if isinstance(name, str) and name.strip():
            rows.append(
                (
                    resolved.canonical_entity_id,
                    resolved.canonical_entity_type,
                    name,
                    normalize_alias(name),
                    "name",
                )
            )
        deduped: list[tuple[str, str, str, str, str]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for row in rows:
            if row in seen:
                continue
            seen.add(row)
            deduped.append(row)
        return deduped

    def _duplicate_rows_from_resolved(
        self,
        resolved: ResolvedCandidate,
        *,
        observed_at: datetime,
        source_turn_id: str | None,
        event_type: str,
    ) -> list[DuplicateCandidate]:
        if resolved.duplicate_target_id is None:
            return []
        return self._duplicate_rows(
            entity_id=resolved.canonical_entity_id,
            candidate_entity_id=resolved.duplicate_target_id,
            observed_at=observed_at,
            source_turn_id=source_turn_id,
            event_type=event_type,
            reason="duplicate entity candidate",
            score=0.75,
            match_basis="ambiguous_alias",
        )

    def _duplicate_rows(
        self,
        *,
        entity_id: str,
        candidate_entity_id: str,
        observed_at: datetime,
        source_turn_id: str | None,
        event_type: str,
        reason: str,
        score: float,
        match_basis: str,
    ) -> list[DuplicateCandidate]:
        return [
            DuplicateCandidate(
                id=str(uuid4()),
                entity_id=entity_id,
                candidate_entity_id=candidate_entity_id,
                match_basis=match_basis,
                score=score,
                status="OPEN",
                reason=reason,
                observed_at=observed_at,
                source_turn_id=source_turn_id,
                event_type=event_type,
            )
        ]


@dataclass(slots=True)
class _ResolvedDraft:
    draft: ExtractedEvent | None = None
    aliases: list[tuple[str, str, str, str, str]] = field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidate] = field(default_factory=list)
    batch_entities: dict[str, ResolvedCandidate] = field(default_factory=dict)


def _union_lists(existing: list[Any], values: list[Any]) -> list[Any]:
    merged = list(existing)
    seen = {repr(item) for item in merged}
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(value)
    return merged
