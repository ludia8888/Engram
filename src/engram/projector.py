from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .storage.store import EventStore
from .types import Entity, RelationEdge


class Projector:
    def __init__(self, store: EventStore):
        self.store = store
        self._snapshot: Mapping[str, Entity] = MappingProxyType({})
        self._relation_snapshot: Mapping[str, tuple[RelationEdge, ...]] = MappingProxyType({})

    def current_snapshot(self) -> Mapping[str, Entity]:
        return self._snapshot

    def current_relation_snapshot(self) -> Mapping[str, tuple[RelationEdge, ...]]:
        return self._relation_snapshot

    def _materialize_owner(self, owner_id: str) -> tuple[Entity | None, tuple[RelationEdge, ...]]:
        entity = self.store.materialize_current_entity(owner_id)
        relations = tuple(self.store.materialize_current_relations(owner_id))
        return entity, relations

    def _apply_owner_materialization(
        self,
        owner_id: str,
        new_snapshot: dict[str, Entity],
        new_relation_snapshot: dict[str, tuple[RelationEdge, ...]],
    ) -> None:
        entity, relations = self._materialize_owner(owner_id)
        if entity is None:
            new_snapshot.pop(owner_id, None)
        else:
            new_snapshot[owner_id] = entity
        if relations:
            new_relation_snapshot[owner_id] = relations
        else:
            new_relation_snapshot.pop(owner_id, None)

    def rebuild_dirty(self) -> int:
        owners = self.store.dirty_owner_ids()
        return self.rebuild_owners(owners)

    def rebuild_owners(self, owner_ids: list[str]) -> int:
        owners = sorted(set(owner_ids))
        if not owners:
            return 0

        new_snapshot = dict(self._snapshot)
        new_relation_snapshot = dict(self._relation_snapshot)
        for owner_id in owners:
            self._apply_owner_materialization(owner_id, new_snapshot, new_relation_snapshot)

        with self.store.transaction() as tx:
            self.store.clear_dirty_ranges_for_owners(tx, owners)

        self._snapshot = MappingProxyType(new_snapshot)
        self._relation_snapshot = MappingProxyType(new_relation_snapshot)
        return len(owners)

    def rebuild_owner(self, owner_id: str, *, related_owner_ids: list[str] | None = None) -> int:
        owners = [owner_id]
        if related_owner_ids:
            owners.extend(related_owner_ids)
        return self.rebuild_owners(owners)

    def rebuild_dirty_until_stable(self) -> int:
        rebuilt_total = 0
        while self.store.dirty_owner_ids():
            rebuilt = self.rebuild_dirty()
            rebuilt_total += rebuilt
            if rebuilt == 0 and self.store.dirty_owner_ids():
                raise RuntimeError("projection rebuild made no progress")
        return rebuilt_total

    def rebuild_all(self) -> int:
        owners = sorted(set(self.store.all_entity_ids()) | set(self._snapshot.keys()) | set(self._relation_snapshot.keys()))
        new_snapshot: dict[str, Entity] = {}
        new_relation_snapshot: dict[str, tuple[RelationEdge, ...]] = {}
        for owner_id in owners:
            self._apply_owner_materialization(owner_id, new_snapshot, new_relation_snapshot)

        with self.store.transaction() as tx:
            self.store.clear_all_dirty_ranges(tx)

        self._snapshot = MappingProxyType(new_snapshot)
        self._relation_snapshot = MappingProxyType(new_relation_snapshot)
        return len(owners)
