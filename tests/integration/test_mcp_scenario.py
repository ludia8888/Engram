"""MCP tool scenario: turn -> recall -> get end-to-end test.

Simulates the user flow:
  1. "나는 채식주의자야" -> engram_turn
  2. "내가 뭘 기억하고 있지?" -> engram_recall
  3. "앨리스 정보 보여줘" -> engram_get
"""
from __future__ import annotations

from engram import Engram
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class ScenarioExtractor:
    """Extractor that produces entity + relation events from a vegetarian turn."""

    version = "scenario-v1"

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        if "채식주의자" in item.user:
            return [
                ExtractedEvent(
                    type="entity.create",
                    data={
                        "id": "user:alice",
                        "type": "user",
                        "attrs": {"name": "앨리스", "diet": "채식주의"},
                    },
                    source_role="user",
                    confidence=0.92,
                    reason="사용자가 본인이 채식주의자라고 말했다",
                    time_confidence="exact",
                ),
            ]
        return []


def test_scenario_turn_then_recall_then_get(tmp_path):
    """Step 1-3: turn -> flush -> recall -> get."""
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=ScenarioExtractor(),
    )

    # --- Step 1: engram_turn ---
    ack = mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 식단 선호로 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
        session_id="sess-001",
    )
    assert ack.turn_id
    assert ack.queued is True

    # flush pipeline: canonical -> projection
    mem.flush("all")

    # --- Step 2: engram_recall ---
    context_text = mem.context(
        "앨리스 채식주의",
        max_tokens=2000,
        include_history=True,
    )
    assert "user:alice" in context_text
    assert "채식주의" in context_text

    # --- Step 3: engram_get ---
    entity = mem.get("user:alice")
    assert entity is not None
    assert entity.type == "user"
    assert entity.attrs["diet"] == "채식주의"
    assert entity.attrs["name"] == "앨리스"

    mem.close()


def test_scenario_search_finds_vegetarian_entity(tmp_path):
    """engram_search: "채식" 검색 시 user:alice가 반환되는지 확인."""
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=ScenarioExtractor(),
    )
    mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.flush("all")

    results = mem.search("채식주의", k=5)
    assert results
    assert results[0].entity_id == "user:alice"

    mem.close()


def test_scenario_history_tracks_diet_change(tmp_path):
    """engram_history: 식단 변경 이력이 기록되는지 확인."""
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=ScenarioExtractor(),
    )
    mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.flush("all")

    # 식단 변경
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"diet": "비건"}},
        observed_at=dt("2026-06-01T10:00:00Z"),
        reason="사용자가 비건으로 전환했다고 말했다",
        confidence=0.95,
    )

    history = mem.known_history("user:alice", attr="diet")
    assert len(history) == 2
    assert history[0].new_value == "채식주의"
    assert history[1].old_value == "채식주의"
    assert history[1].new_value == "비건"

    mem.close()


def test_scenario_get_returns_none_for_unknown_entity(tmp_path):
    """engram_get: 존재하지 않는 entity 조회 시 None 반환."""
    mem = Engram(user_id="alice", path=str(tmp_path))

    assert mem.get("user:unknown") is None

    mem.close()


def test_scenario_recall_with_raw_evidence(tmp_path):
    """engram_recall(include_raw=True): 원본 대화가 증거로 포함되는지 확인."""
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=ScenarioExtractor(),
    )
    ack = mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 식단 선호로 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.flush("all")

    # source_turn_id를 연결해서 append
    mem.append(
        "entity.update",
        {"id": "user:alice", "attrs": {"location": "부산"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        source_turn_id=ack.turn_id,
        reason="사용자가 부산에 산다고 말했다",
        confidence=0.9,
    )

    context_text = mem.context(
        "부산 채식 식당",
        max_tokens=2000,
        include_raw=True,
    )
    assert "부산" in context_text
    assert "채식주의" in context_text
    assert "## Raw Evidence" in context_text
    assert "나는 채식주의자야" in context_text

    mem.close()


def test_scenario_relations_between_entities(tmp_path):
    """engram_get_relations: entity 간 관계 조회."""
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=ScenarioExtractor(),
    )
    mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )
    mem.flush("all")

    # 프로젝트 entity 생성 + 관계 추가
    mem.append(
        "entity.create",
        {"id": "project:cooking", "type": "project", "attrs": {"name": "채식 요리 프로젝트"}},
        observed_at=dt("2026-05-02T10:00:00Z"),
    )
    mem.append(
        "relation.create",
        {
            "source": "user:alice",
            "target": "project:cooking",
            "type": "member",
            "attrs": {"role": "owner"},
        },
        observed_at=dt("2026-05-02T10:01:00Z"),
    )

    edges = mem.get_relations("user:alice")
    assert len(edges) == 1
    assert edges[0].other_entity_id == "project:cooking"
    assert edges[0].relation_type == "member"
    assert edges[0].attrs["role"] == "owner"

    mem.close()


def test_scenario_flush_then_get_shows_latest_state(tmp_path):
    """engram_flush -> engram_get: flush 후 최신 상태 반영 확인."""
    mem = Engram(
        user_id="alice",
        path=str(tmp_path),
        extractor=ScenarioExtractor(),
    )
    mem.turn(
        user="나는 채식주의자야",
        assistant="좋아, 기억해둘게.",
        observed_at=dt("2026-05-01T10:00:00Z"),
    )

    # flush 전에는 canonical event가 없으므로 get은 None
    assert mem.get("user:alice") is None

    mem.flush("all")

    # flush 후 entity가 보여야 함
    entity = mem.get("user:alice")
    assert entity is not None
    assert entity.attrs["diet"] == "채식주의"

    mem.close()
