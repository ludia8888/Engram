from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .storage.store import EventStore
from .types import Entity


class Projector:
    def __init__(self, store: EventStore):
        self.store = store
        self._snapshot: Mapping[str, Entity] = MappingProxyType({})

    def current_snapshot(self) -> Mapping[str, Entity]:
        return self._snapshot

    def rebuild_dirty(self) -> int:
        owners = self.store.dirty_owner_ids()
        if not owners:
            return 0

        new_snapshot = dict(self._snapshot)
        for owner_id in owners:
            entity = self.store.materialize_current_entity(owner_id)
            if entity is None:
                new_snapshot.pop(owner_id, None)
            else:
                new_snapshot[owner_id] = entity

        with self.store.transaction() as tx:
            self.store.clear_dirty_ranges_for_owners(tx, owners)

        self._snapshot = MappingProxyType(new_snapshot)
        return len(owners)

    def rebuild_all(self) -> int:
        owners = self.store.all_entity_ids()
        new_snapshot: dict[str, Entity] = {}
        for owner_id in owners:
            entity = self.store.materialize_current_entity(owner_id)
            if entity is not None:
                new_snapshot[owner_id] = entity

        with self.store.transaction() as tx:
            self.store.clear_all_dirty_ranges(tx)

        self._snapshot = MappingProxyType(new_snapshot)
        return len(new_snapshot)
