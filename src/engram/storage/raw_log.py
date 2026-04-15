from __future__ import annotations

import json
import os
import threading
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
            return TurnAck(
                turn_id=turn.id,
                observed_at=turn.observed_at,
                durable_at=durable_at,
                queued=True,
            )

    def raw_get(self, turn_id: str) -> RawTurn | None:
        for turn in self._iter_turns():
            if turn.id == turn_id:
                return turn
        return None

    def raw_recent(self, limit: int = 20) -> list[RawTurn]:
        turns = list(self._iter_turns())
        return list(reversed(turns[-limit:]))

    def _iter_turns(self):
        manifest = self._load_manifest()
        active = self.root / manifest["active_segment"]
        paths = sorted(self.archived.glob("*.jsonl.gz"))
        if active.exists():
            paths.append(active)
        for path in paths:
            if path.suffix == ".gz":
                continue
            with path.open("r", encoding="utf-8") as handle:
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

