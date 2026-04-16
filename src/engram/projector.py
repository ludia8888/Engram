from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from .snapshot_serde import deserialize_snapshot, serialize_snapshot
from .storage.store import EventStore
from .time_utils import from_rfc3339, utcnow
from .types import Entity, RelationEdge, SnapshotRow


class Projector:
    def __init__(self, store: EventStore):
        self.store = store
        self._snapshot: Mapping[str, Entity] = MappingProxyType({})
        self._relation_snapshot: Mapping[str, tuple[RelationEdge, ...]] = MappingProxyType({})
        self._projection_version: int = 0
        self._snapshot_last_seq: int = 0

    def current_snapshot(self) -> Mapping[str, Entity]:
        return self._snapshot

    @property
    def snapshot_last_seq(self) -> int:
        return self._snapshot_last_seq

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
        self._projection_version += 1
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
        self._projection_version += 1
        return len(owners)

    def save_snapshot(self) -> str | None:
        if not self._snapshot and not self._relation_snapshot:
            return None

        last_seq = self.store.current_max_seq()
        if last_seq == 0:
            return None

        max_recorded_at_str = self.store.max_recorded_at_for_seq(last_seq)
        if max_recorded_at_str is None:
            return None

        state_blob, relation_blob = serialize_snapshot(self._snapshot, self._relation_snapshot)
        snapshot_id = str(uuid4())
        row = SnapshotRow(
            id=snapshot_id,
            basis="known",
            created_at=utcnow(),
            last_seq=last_seq,
            projection_version=self._projection_version,
            max_recorded_at_included=from_rfc3339(max_recorded_at_str),
            max_effective_at_included=None,
            state_blob=state_blob,
            relation_blob=relation_blob,
        )
        with self.store.transaction() as tx:
            self.store.save_snapshot(tx, row)
            self.store.delete_old_snapshots(tx, keep_count=3)
        return snapshot_id

    def load_snapshot(self) -> bool:
        row = self.store.load_latest_snapshot()
        if row is None:
            return False
        try:
            entities, relations = deserialize_snapshot(row.state_blob, row.relation_blob)
        except Exception:
            with self.store.transaction() as tx:
                self.store.delete_snapshot_by_id(tx, row.id)
            return False
        self._snapshot = MappingProxyType(entities)
        self._relation_snapshot = MappingProxyType(relations)
        self._projection_version = row.projection_version
        self._snapshot_last_seq = row.last_seq
        return True
