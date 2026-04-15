from __future__ import annotations

import pytest

from engram import Engram, ValidationError
import engram.engram as engram_module

from tests.conftest import dt


def test_search_returns_entity_seeded_by_matching_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="user said they are vegetarian",
        confidence=0.91,
        source_role="user",
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
        reason="user said they moved to Busan",
        confidence=0.95,
        source_role="user",
        time_confidence="exact",
    )
    mem.append(
        "entity.create",
        {"id": "project:trip", "type": "project", "attrs": {"destination": "Tokyo"}},
        observed_at=dt("2026-05-01T12:00:00Z"),
        reason="trip planning started",
        confidence=0.7,
        source_role="manual",
        time_confidence="exact",
    )

    results = mem.search("Busan vegetarian", k=5)

    assert results
    assert results[0].entity_id == "user:alice"
    assert results[0].time_basis == "known"
    assert "entity" in results[0].matched_axes
    assert len(results[0].supporting_event_ids) >= 2

    mem.close()


def test_search_respects_known_time_window(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-01T10:00:00Z"))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="initial city",
        source_role="manual",
    )
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-10T10:00:00Z"))
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        reason="moved to Busan",
        source_role="manual",
    )

    early = mem.search(
        "Busan",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-05T00:00:00Z")),
    )
    late = mem.search(
        "Busan",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
    )

    assert early == []
    assert late[0].entity_id == "user:alice"
    assert "temporal" in late[0].matched_axes

    mem.close()


def test_context_builds_known_time_memory_summary_and_optional_raw(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    ack = mem.turn(
        user="지난주에 부산으로 이사했고 나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "채식주의"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="사용자가 본인이 채식주의자라고 말했다",
        confidence=0.91,
        source_role="user",
        source_turn_id=ack.turn_id,
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "부산"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="사용자가 지난주에 부산으로 이사했다고 말했다",
        confidence=0.95,
        source_role="user",
        source_turn_id=ack.turn_id,
        time_confidence="inferred",
    )

    text = mem.context(
        "부산에서 채식 식당 추천해줘",
        max_tokens=400,
        include_raw=True,
    )

    assert "## Memory Basis" in text
    assert "## Current State" in text
    assert "## Relevant Changes" in text
    assert "## Raw Evidence" in text
    assert "user:alice" in text
    assert "부산" in text
    assert "채식주의" in text
    assert "지난주에 부산으로 이사했고 나는 채식주의자야" in text

    mem.close()


def test_search_and_context_reject_valid_mode_for_now(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))

    with pytest.raises(ValidationError, match="valid time_mode is planned"):
        mem.search("Busan", time_mode="valid")

    with pytest.raises(ValidationError, match="valid time_mode is planned"):
        mem.context("Busan", time_mode="valid")

    mem.close()
