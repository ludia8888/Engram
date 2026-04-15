from __future__ import annotations

import gzip
import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from engram.time_utils import from_rfc3339, to_rfc3339, utcnow
from engram.types import RawTurn, TurnAck


class SegmentedRawLog:
    def __init__(self, root: Path):
        self.root = root
        self.archived = root / "archived"
        self.manifest_path = root / "manifest.json"
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.archived.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest(
                {
                    "active_segment": "active-000001.jsonl",
                    "last_committed_turn_id": None,
                    "last_rotation_at": to_rfc3339(utcnow()),
                }
            )
        self._index: dict[str, RawTurn] | None = None
        self._all_turn_ids: list[str] | None = None
        self._session_index: dict[str | None, list[str]] | None = None

    def append(self, turn: RawTurn) -> TurnAck:
        with self._lock:
            manifest = self._load_manifest()
            segment_path = self.root / manifest["active_segment"]
            record = {
                "id": turn.id,
                "session_id": turn.session_id,
                "observed_at": to_rfc3339(turn.observed_at),
                "user": turn.user,
                "assistant": turn.assistant,
                "metadata": turn.metadata,
            }
            with segment_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            durable_at = utcnow()
            manifest["last_committed_turn_id"] = turn.id
            self._write_manifest(manifest)
            if self._index is not None:
                self._index[turn.id] = turn
                self._all_turn_ids.append(turn.id)
                self._session_index.setdefault(turn.session_id, []).append(turn.id)
            return TurnAck(
                turn_id=turn.id,
                observed_at=turn.observed_at,
                durable_at=durable_at,
                queued=True,
            )

    def raw_get(self, turn_id: str) -> RawTurn | None:
        self._ensure_index()
        return self._index.get(turn_id)

    def raw_recent(self, limit: int = 20) -> list[RawTurn]:
        self._ensure_index()
        ids = self._all_turn_ids[-limit:]
        return [self._index[tid] for tid in reversed(ids)]

    def raw_recent_for_session(self, session_id: str, limit: int = 20) -> list[RawTurn]:
        self._ensure_index()
        ids = self._session_index.get(session_id, [])[-limit:]
        return [self._index[tid] for tid in reversed(ids)]

    def raw_all(self) -> Iterator[RawTurn]:
        return self._iter_turns()

    def raw_range(
        self,
        *,
        from_turn_id: str | None = None,
        to_turn_id: str | None = None,
    ) -> list[RawTurn]:
        self._ensure_index()
        all_ids = self._all_turn_ids
        start_index = 0
        end_index = len(all_ids) - 1

        if from_turn_id is not None:
            if from_turn_id not in self._index:
                raise KeyError(from_turn_id)
            start_index = all_ids.index(from_turn_id)
        if to_turn_id is not None:
            if to_turn_id not in self._index:
                raise KeyError(to_turn_id)
            end_index = all_ids.index(to_turn_id)

        if start_index > end_index:
            raise ValueError("from_turn_id_after_to_turn_id")

        return [self._index[tid] for tid in all_ids[start_index : end_index + 1]]

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        index: dict[str, RawTurn] = {}
        all_ids: list[str] = []
        session_index: dict[str | None, list[str]] = defaultdict(list)
        for turn in self._iter_turns():
            index[turn.id] = turn
            all_ids.append(turn.id)
            session_index[turn.session_id].append(turn.id)
        self._index = index
        self._all_turn_ids = all_ids
        self._session_index = dict(session_index)

    def _iter_turns(self):
        manifest = self._load_manifest()
        active = self.root / manifest["active_segment"]
        paths = sorted(self.archived.glob("*.jsonl.gz"))
        if active.exists():
            paths.append(active)
        for path in paths:
            if path.suffix == ".gz":
                handle = gzip.open(path, "rt", encoding="utf-8")
            else:
                handle = path.open("r", encoding="utf-8")
            with handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    yield RawTurn(
                        id=data["id"],
                        session_id=data.get("session_id"),
                        observed_at=from_rfc3339(data["observed_at"]),
                        user=data["user"],
                        assistant=data["assistant"],
                        metadata=data.get("metadata") or {},
                    )

    def _load_manifest(self) -> dict:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_manifest(self, manifest: dict) -> None:
        temp_path = self.manifest_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self.manifest_path)
