from __future__ import annotations

from tempfile import TemporaryDirectory

from hypothesis import given
from hypothesis import strategies as st

from engram import Engram


@given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=8))
def test_seq_is_monotonic_and_current_state_matches_last_history(values):
    with TemporaryDirectory() as tmpdir:
        mem = Engram(user_id="alice", path=tmpdir)
        mem.append(
            "entity.create",
            {"id": "user:alice", "type": "user", "attrs": {"status": values[0]}},
        )
        for value in values[1:]:
            mem.append(
                "entity.update",
                {"id": "user:alice", "attrs": {"status": value}},
            )

        seqs = [row[0] for row in mem.conn.execute("SELECT seq FROM events ORDER BY seq").fetchall()]
        assert seqs == list(range(1, len(seqs) + 1))

        current = mem.get("user:alice")
        history = mem.known_history("user:alice", attr="status")
        assert current is not None
        assert history[-1].new_value == current.attrs["status"]
        mem.close()
