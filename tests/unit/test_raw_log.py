from __future__ import annotations

import json
from uuid import uuid4

from engram.storage.raw_log import SegmentedRawLog
from engram.types import RawTurn

from tests.conftest import dt


def test_segmented_raw_log_updates_manifest_and_reads_recent(tmp_path):
    raw_log = SegmentedRawLog(tmp_path / "raw")
    turn = RawTurn(
        id=str(uuid4()),
        session_id="session-1",
        observed_at=dt("2026-05-01T10:00:00Z"),
        user="u1",
        assistant="a1",
        metadata={"source": "chat"},
    )

    ack = raw_log.append(turn)
    manifest = json.loads((tmp_path / "raw" / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["last_committed_turn_id"] == turn.id
    assert ack.turn_id == turn.id
    assert raw_log.raw_get(turn.id) is not None
    assert raw_log.raw_recent(1)[0].id == turn.id

