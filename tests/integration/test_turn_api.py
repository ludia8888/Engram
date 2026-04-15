from __future__ import annotations

import json
import sqlite3

from engram import Engram

from tests.conftest import dt


def test_turn_returns_ack_and_only_touches_raw_layer(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    observed_at = dt("2026-05-01T10:00:00Z")

    ack = mem.turn(
        user="지난주에 부산으로 이사했어",
        assistant="알겠어, 부산 기준으로 추천할게.",
        observed_at=observed_at,
        metadata={"source": "chat"},
    )

    assert ack.turn_id
    assert ack.observed_at == observed_at
    assert ack.queued is True

    user_root = tmp_path / "alice"
    manifest = json.loads((user_root / "raw" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_segment"] == "active-000001.jsonl"
    assert manifest["last_committed_turn_id"] == ack.turn_id

    stored_turn = mem.raw_get(ack.turn_id)
    assert stored_turn is not None
    assert stored_turn.user == "지난주에 부산으로 이사했어"
    assert mem.raw_recent(1)[0].id == ack.turn_id
    assert mem.queue.qsize() == 1

    with sqlite3.connect(mem.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0

    mem.close()

