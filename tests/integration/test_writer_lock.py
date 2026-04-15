from __future__ import annotations

import pytest

from engram import Engram, WriterLockError


def test_same_storage_root_rejects_second_writer(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    with pytest.raises(WriterLockError):
        Engram(user_id="alice", path=str(tmp_path))
    mem.close()

