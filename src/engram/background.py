from __future__ import annotations

import heapq
import logging
import queue
import threading
import time

from .canonical import CanonicalWorker
from .projector import Projector
from .retry import RetryPolicy
from .semantic_index import SemanticIndexer
from .types import QueueItem

logger = logging.getLogger(__name__)


class BackgroundWorker:
    def __init__(
        self,
        work_queue: queue.Queue[QueueItem],
        canonical_worker: CanonicalWorker,
        projector: Projector,
        semantic_indexer: SemanticIndexer,
        retry_policy: RetryPolicy,
        drain_timeout: float = 0.5,
    ):
        self._queue = work_queue
        self._canonical = canonical_worker
        self._projector = projector
        self._indexer = semantic_indexer
        self._retry_policy = retry_policy
        self._drain_timeout = drain_timeout
        self._stop_event = threading.Event()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._retry_heap: list[tuple[float, str, QueueItem]] = []
        self._retry_counter = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="engram-background-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._cond:
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def notify(self) -> None:
        with self._cond:
            self._cond.notify()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            processed_any = self._drain_and_process()
            if processed_any:
                self._rebuild_and_index()
            with self._cond:
                self._cond.wait(timeout=self._drain_timeout)

    def _drain_and_process(self) -> bool:
        processed = False
        now = time.monotonic()
        while self._retry_heap and self._retry_heap[0][0] <= now:
            _, _, item = heapq.heappop(self._retry_heap)
            self._process_one(item)
            processed = True
        while not self._stop_event.is_set():
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._process_one(item)
            self._queue.task_done()
            processed = True
        return processed

    def _process_one(self, item: QueueItem) -> None:
        try:
            success, retry_state = self._canonical.process_with_retry(
                item, self._retry_policy,
            )
            if not success and retry_state is not None:
                self._retry_counter += 1
                heapq.heappush(
                    self._retry_heap,
                    (retry_state.next_eligible_at, self._retry_counter, item),
                )
        except Exception:
            logger.exception("unexpected error processing turn %s", item.turn_id)

    def _rebuild_and_index(self) -> None:
        try:
            self._projector.rebuild_dirty_until_stable()
            self._projector.save_snapshot()
        except Exception:
            logger.exception("projection rebuild failed")
        try:
            self._indexer.index_missing()
        except Exception:
            logger.exception("semantic indexing failed")
