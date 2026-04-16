from __future__ import annotations

import queue

from .errors import QueueFullError
from .meaning_index import MeaningIndexer
from .projector import Projector
from .semantic_index import SemanticIndexer
from .storage.raw_log import SegmentedRawLog
from .storage.store import EventStore
from .types import QueueItem


class RecoveryService:
    def __init__(
        self,
        raw_log: SegmentedRawLog,
        store: EventStore,
        projector: Projector,
        semantic_indexer: SemanticIndexer,
        meaning_indexer: MeaningIndexer,
        work_queue: queue.Queue[QueueItem],
        queue_put_timeout: float,
        extractor_version: str,
    ):
        self.raw_log = raw_log
        self.store = store
        self.projector = projector
        self.semantic_indexer = semantic_indexer
        self.meaning_indexer = meaning_indexer
        self.work_queue = work_queue
        self.queue_put_timeout = queue_put_timeout
        self.extractor_version = extractor_version

    def catch_up_on_startup(self) -> int:
        snapshot_loaded = self.projector.load_snapshot()
        canonical_max_seq = self.store.current_max_seq()

        if snapshot_loaded and self.projector.snapshot_last_seq < canonical_max_seq:
            self.projector.rebuild_all()
        elif not snapshot_loaded and canonical_max_seq > 0:
            self.projector.rebuild_all()
        else:
            while self.store.count_dirty_ranges() > 0:
                rebuilt = self.projector.rebuild_dirty()
                if rebuilt == 0 and self.store.count_dirty_ranges() > 0:
                    raise RuntimeError("startup projection recovery made no progress")

        self.semantic_indexer.index_missing()
        self.meaning_indexer.index_missing()

        processed_turn_ids = self.store.successful_source_turn_ids(self.extractor_version)
        enqueued = 0
        for turn in self.raw_log.raw_all():
            if turn.id in processed_turn_ids:
                continue
            try:
                self.work_queue.put(QueueItem.from_turn(turn), timeout=self.queue_put_timeout)
            except queue.Full as exc:
                raise QueueFullError(
                    f"startup catch-up could not enqueue raw turn {turn.id}"
                ) from exc
            enqueued += 1
        return enqueued
