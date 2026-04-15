from __future__ import annotations

from engram import Engram
import engram.engram as engram_module
import engram.retrieval as retrieval_module

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
    moved_event_id = mem.append(
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
    assert moved_event_id in late[0].supporting_event_ids

    mem.close()


def test_search_normalizes_korean_particles_for_lexical_match(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    monkeypatch.setattr(engram_module, "utcnow", lambda: dt("2026-05-10T10:00:00Z"))
    moved_event_id = mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "부산"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        reason="사용자가 부산으로 이사했다고 말했다",
        source_role="user",
    )

    results = mem.search(
        "부산에서 뭐 먹지",
        k=5,
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
    )

    assert results
    assert results[0].entity_id == "user:alice"
    assert moved_event_id in results[0].supporting_event_ids

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


def test_search_respects_valid_time_window(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    moved_event_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
        reason="moved to Busan",
    )

    early = mem.search(
        "Busan",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-05T00:00:00Z")),
    )
    late = mem.search(
        "Busan",
        time_mode="valid",
        time_window=(dt("2026-05-08T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
    )

    assert early == []
    assert late[0].entity_id == "user:alice"
    assert late[0].time_basis == "valid"
    assert "temporal" in late[0].matched_axes
    assert moved_event_id in late[0].supporting_event_ids

    mem.close()


def test_search_valid_skips_unknown_effective_time_events(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        time_confidence="unknown",
        reason="location mentioned without exact effective time",
    )

    results = mem.search("Busan", time_mode="valid")

    assert results == []

    mem.close()


def test_search_valid_without_time_window_uses_current_as_of(tmp_path, monkeypatch):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    moved_event_id = mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-10T10:00:00Z"),
        effective_at_start=dt("2026-05-08T00:00:00Z"),
        time_confidence="exact",
        reason="moved to Busan",
    )

    monkeypatch.setattr(retrieval_module, "utcnow", lambda: dt("2026-05-09T12:00:00Z"))

    results = mem.search("Busan", time_mode="valid")

    assert results
    assert results[0].entity_id == "user:alice"
    assert results[0].time_basis == "valid"
    assert moved_event_id in results[0].supporting_event_ids

    mem.close()


def test_context_builds_valid_time_memory_summary(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    ack = mem.turn(
        user="나는 채식주의자고, 부산에 있는 것 같긴 한데 정확한 날짜는 잘 모르겠어",
        assistant="좋아, 시간 확실한 정보와 불확실한 정보를 나눠서 기억해둘게.",
        observed_at=dt("2026-05-12T09:00:00Z"),
    )
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"diet": "vegetarian"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        source_turn_id=ack.turn_id,
        source_role="user",
        time_confidence="exact",
        reason="diet is known to be active from May 1",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-12T09:00:00Z"),
        source_turn_id=ack.turn_id,
        source_role="user",
        time_confidence="unknown",
        reason="location was mentioned but effective time is unknown",
    )

    text = mem.context(
        "vegetarian meal ideas",
        time_mode="valid",
        time_window=(dt("2026-05-01T00:00:00Z"), dt("2026-05-15T00:00:00Z")),
        include_raw=True,
        max_tokens=400,
    )

    assert "## Memory Basis" in text
    assert "- mode: valid" in text
    assert "## Current State" in text
    assert "user:alice" in text
    assert "vegetarian" in text
    assert "unknown_attrs_as_of_window_end=['location']" in text
    assert "## Relevant Changes" in text
    assert "effective_at=2026-05-01T00:00:00Z" in text
    assert "## Raw Evidence" in text
    assert "나는 채식주의자고" in text

    mem.close()


def test_batch_known_helpers_match_single_entity_reads(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="moved to Busan",
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
    )
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-01T10:06:00Z"),
    )

    at = dt("2026-05-02T00:00:00Z")
    views = mem._get_known_views_at_many(["user:alice", "user:bob"], at)
    relations = mem._get_known_relations_at_many(["user:alice", "user:bob"], at)

    assert views["user:alice"] == mem.get_known_at("user:alice", at)
    assert views["user:bob"] == mem.get_known_at("user:bob", at)
    assert relations["user:alice"] == mem.get_relations("user:alice", time_mode="known", at=at)
    assert relations["user:bob"] == mem.get_relations("user:bob", time_mode="known", at=at)

    mem.close()


def test_batch_valid_helpers_match_single_entity_reads(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "user:alice", "type": "user", "attrs": {"location": "Seoul"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "Busan"}},
        observed_at=dt("2026-05-05T10:00:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "entity.create",
        {"id": "user:bob", "type": "user", "attrs": {"team": "engram"}},
        observed_at=dt("2026-05-01T10:05:00Z"),
        effective_at_start=dt("2026-05-01T00:00:00Z"),
        time_confidence="exact",
    )
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "user:bob",
            "type": "manager",
            "attrs": {"scope": "engram"},
        },
        observed_at=dt("2026-05-05T10:06:00Z"),
        effective_at_start=dt("2026-05-05T00:00:00Z"),
        time_confidence="exact",
    )

    at = dt("2026-05-06T00:00:00Z")
    views = mem._get_valid_views_at_many(["user:alice", "user:bob"], at)
    relations = mem._get_valid_relations_at_many(["user:alice", "user:bob"], at)

    assert views["user:alice"] == mem.get_valid_at("user:alice", at)
    assert views["user:bob"] == mem.get_valid_at("user:bob", at)
    assert relations["user:alice"] == mem.get_relations("user:alice", time_mode="valid", at=at)
    assert relations["user:bob"] == mem.get_relations("user:bob", time_mode="valid", at=at)

    window_relations = mem._get_valid_relations_in_window_many(
        ["user:alice", "user:bob"],
        dt("2026-05-04T00:00:00Z"),
        dt("2026-05-07T00:00:00Z"),
    )

    assert window_relations["user:alice"] == mem.get_relations(
        "user:alice",
        time_mode="valid",
        time_window=(dt("2026-05-04T00:00:00Z"), dt("2026-05-07T00:00:00Z")),
    )
    assert window_relations["user:bob"] == mem.get_relations(
        "user:bob",
        time_mode="valid",
        time_window=(dt("2026-05-04T00:00:00Z"), dt("2026-05-07T00:00:00Z")),
    )

    mem.close()
