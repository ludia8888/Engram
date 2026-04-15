from __future__ import annotations

import json

import pytest

from engram import Engram, WriterLockError
from engram.storage.locks import WriterLock


def test_same_storage_root_rejects_second_writer(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    with pytest.raises(WriterLockError):
        Engram(user_id="alice", path=str(tmp_path))
    mem.close()


def test_stale_writer_lock_is_recovered_when_pid_is_not_alive(tmp_path, monkeypatch):
    user_root = tmp_path / "alice"
    user_root.mkdir(parents=True, exist_ok=True)
    lock_path = user_root / ".writer.lock"
    lock_path.write_text(
        json.dumps({"pid": 424242, "created_at": "2026-04-15T12:00:00Z"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(WriterLock, "_pid_is_alive", staticmethod(lambda pid: False))

    mem = Engram(user_id="alice", path=str(tmp_path))
    assert mem._writer_lock.path.exists()
    mem.close()


def test_legacy_pid_only_lockfile_is_also_recovered_if_pid_is_dead(tmp_path, monkeypatch):
    user_root = tmp_path / "alice"
    user_root.mkdir(parents=True, exist_ok=True)
    lock_path = user_root / ".writer.lock"
    lock_path.write_text("424242", encoding="utf-8")

    monkeypatch.setattr(WriterLock, "_pid_is_alive", staticmethod(lambda pid: False))

    mem = Engram(user_id="alice", path=str(tmp_path))
    assert mem._writer_lock.path.exists()
    mem.close()
