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

_SLOW_THRESHOLD_SECS = 2.0


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
        self._retry_heap: list[tuple[float, int, QueueItem]] = []
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
        processed_count = 0
        retry_count = 0
        now = time.monotonic()
        while self._retry_heap and self._retry_heap[0][0] <= now:
            _, _, item = heapq.heappop(self._retry_heap)
            self._process_one(item)
            retry_count += 1
        while not self._stop_event.is_set():
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._process_one(item)
            self._queue.task_done()
            processed_count += 1
        if processed_count > 0 or retry_count > 0:
            logger.debug("drain: processed=%d retried=%d", processed_count, retry_count)
        return (processed_count + retry_count) > 0

    def _process_one(self, item: QueueItem) -> None:
        try:
            t0 = time.monotonic()
            success, retry_state = self._canonical.process_with_retry(
                item, self._retry_policy,
            )
            elapsed = time.monotonic() - t0
            if success:
                logger.info("canonical: turn=%s processed in %.3fs", item.turn_id, elapsed)
            if not success and retry_state is not None:
                self._retry_counter += 1
                heapq.heappush(
                    self._retry_heap,
                    (retry_state.next_eligible_at, self._retry_counter, item),
                )
                logger.warning(
                    "canonical: turn=%s scheduled retry attempt=%d delay=%.1fs",
                    item.turn_id,
                    retry_state.attempt,
                    retry_state.next_eligible_at - time.monotonic(),
                )
        except Exception:
            logger.exception("unexpected error processing turn %s", item.turn_id)

    def _rebuild_and_index(self) -> None:
        try:
            t0 = time.monotonic()
            rebuilt = self._projector.rebuild_dirty_until_stable()
            t1 = time.monotonic()
            snapshot_id = self._projector.save_snapshot()
            t2 = time.monotonic()
            total = t2 - t0
            logger.info(
                "projection: rebuilt=%d owners in %.3fs, snapshot=%s in %.3fs",
                rebuilt, t1 - t0, snapshot_id or "skipped", t2 - t1,
            )
            if total > _SLOW_THRESHOLD_SECS:
                logger.warning("slow projection cycle: %.3fs", total)
        except Exception:
            logger.exception("projection rebuild failed")
        try:
            t0 = time.monotonic()
            indexed = self._indexer.index_missing()
            elapsed = time.monotonic() - t0
            if indexed > 0:
                logger.info("semantic index: indexed=%d in %.3fs", indexed, elapsed)
        except Exception:
            logger.exception("semantic indexing failed")
