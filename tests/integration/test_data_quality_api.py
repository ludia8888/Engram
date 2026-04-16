from __future__ import annotations

from engram import Engram
from engram.schema_registry import normalize_alias
from engram.types import ExtractedEvent, QueueItem

from tests.conftest import dt


class QueueDrivenExtractor:
    version = "queue-driven-v1"

    def __init__(self, batches: list[list[ExtractedEvent]]):
        self._batches = list(batches)

    def extract(self, item: QueueItem) -> list[ExtractedEvent]:
        if not self._batches:
            return []
        return self._batches.pop(0)


def test_auto_extraction_reuses_existing_manual_person_entity(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "person:sujin", "type": "person", "attrs": {"name": "수진", "origin": "부산"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="manual seed",
    )
    mem.close()

    extractor = QueueDrivenExtractor(
        [
            [
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "person:수진", "type": "person", "attrs": {"name": "수진"}},
                    reason="auto mention",
                    confidence=0.9,
                )
            ]
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    mem.turn(
        user="수진이가 합류했어",
        assistant="좋아.",
        observed_at=dt("2026-05-02T10:00:00Z"),
    )
    mem.flush("all")

    entity = mem.get("person:sujin")
    assert entity is not None
    assert entity.id == "person:sujin"
    assert mem.store.entity_exists("person:수진") is False
    results = mem.search("수진", k=5)
    assert results
    assert [row.entity_id for row in results] == ["person:sujin"]

    mem.close()


def test_existing_project_status_is_updated_instead_of_creating_new_entity(tmp_path):
    extractor = QueueDrivenExtractor(
        [
            [
                ExtractedEvent(
                    type="entity.update",
                    data={"id": "entity:project-alpha", "attrs": {"status": "완료"}},
                    reason="status changed",
                    confidence=0.92,
                )
            ]
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    mem.append(
        "entity.create",
        {"id": "project:alpha", "type": "project", "attrs": {"name": "Project Alpha", "status": "review"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="manual seed",
    )
    mem.turn(
        user="알파 프로젝트 완료됐어",
        assistant="좋아.",
        observed_at=dt("2026-05-02T10:00:00Z"),
    )
    mem.flush("all")

    entity = mem.get("project:alpha")
    assert entity is not None
    assert entity.attrs["status"] == "done"
    history = mem.known_history("project:alpha", attr="status")
    assert [entry.new_value for entry in history][-1] == "done"
    assert mem.store.entity_exists("entity:project-alpha") is False

    mem.close()


def test_auto_created_project_id_uses_project_prefix(tmp_path):
    extractor = QueueDrivenExtractor(
        [
            [
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "Project Beta", "type": "project", "attrs": {"name": "Project Beta"}},
                    reason="new project",
                    confidence=0.93,
                )
            ]
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    mem.turn(
        user="Project Beta 맡았어",
        assistant="좋아.",
        observed_at=dt("2026-05-02T10:00:00Z"),
    )
    mem.flush("all")

    entity = mem.get("project:beta")
    assert entity is not None
    assert entity.type == "project"
    assert mem.store.entity_exists("entity:project-beta") is False

    mem.close()


def test_duplicate_candidate_is_recorded_for_ambiguous_alias(tmp_path):
    extractor = QueueDrivenExtractor(
        [
            [
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "박수진", "type": "person", "attrs": {"name": "박수진"}},
                    reason="ambiguous person mention",
                    confidence=0.88,
                )
            ]
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    mem.append(
        "entity.create",
        {"id": "person:sujin", "type": "person", "attrs": {"name": "수진"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="manual seed 1",
    )
    mem.append(
        "entity.create",
        {"id": "person:park-sujin", "type": "person", "attrs": {"name": "수진(외부)"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
        reason="manual seed 2",
    )
    with mem.store.transaction() as tx:
        mem.store.append_entity_alias_rows(
            tx,
            [
                (
                    "person:sujin",
                    "person",
                    "박수진",
                    normalize_alias("박수진"),
                    "alias",
                    dt("2026-05-01T12:00:00Z"),
                ),
                (
                    "person:park-sujin",
                    "person",
                    "박수진",
                    normalize_alias("박수진"),
                    "alias",
                    dt("2026-05-01T12:00:00Z"),
                ),
            ],
        )
    mem.turn(
        user="박수진이 합류했어",
        assistant="좋아.",
        observed_at=dt("2026-05-02T10:00:00Z"),
    )
    mem.flush("all")

    rows = mem.list_duplicate_candidates(limit=10)
    assert rows
    assert rows[0].status == "OPEN"
    context = mem.context("박수진", max_tokens=1200)
    assert "Duplicate Hints" in context

    mem.close()


def test_merge_entities_redirects_get_and_search_to_canonical_entity(tmp_path):
    mem = Engram(user_id="alice", path=str(tmp_path))
    mem.append(
        "entity.create",
        {"id": "person:sujin", "type": "person", "attrs": {"name": "수진"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="manual seed",
    )
    mem.append(
        "entity.create",
        {"id": "person:수진", "type": "person", "attrs": {"name": "수진", "origin": "부산"}},
        observed_at=dt("2026-05-01T11:00:00Z"),
        reason="duplicate seed",
    )

    merged_to = mem.merge_entities("person:수진", "person:sujin", reason="same real-world person")
    assert merged_to == "person:sujin"

    source_lookup = mem.get("person:수진")
    assert source_lookup is not None
    assert source_lookup.id == "person:sujin"
    assert "person:수진" in source_lookup.redirected_from

    results = mem.search("부산 출신", k=5)
    assert results
    assert [row.entity_id for row in results] == ["person:sujin"]

    mem.close()


def test_manual_and_auto_paths_share_relation_and_id_rules(tmp_path):
    extractor = QueueDrivenExtractor(
        [
            [
                ExtractedEvent(
                    type="entity.create",
                    data={"id": "Project Beta", "type": "project", "attrs": {"name": "Project Beta"}},
                    reason="project mention",
                    confidence=0.95,
                ),
                ExtractedEvent(
                    type="relation.create",
                    data={
                        "source": "person:sujin",
                        "target": "Project Beta",
                        "type": "works_on",
                        "attrs": {"role": "UI 디자이너"},
                    },
                    reason="role mention",
                    confidence=0.95,
                ),
            ]
        ]
    )
    mem = Engram(user_id="alice", path=str(tmp_path), extractor=extractor)
    mem.append(
        "entity.create",
        {"id": "person:sujin", "type": "person", "attrs": {"name": "수진"}},
        observed_at=dt("2026-05-01T10:00:00Z"),
        reason="manual person",
    )
    mem.turn(
        user="수진이 UI 디자이너로 Project Beta 맡았어",
        assistant="좋아.",
        observed_at=dt("2026-05-02T10:00:00Z"),
    )
    mem.flush("all")

    project = mem.get("project:beta")
    assert project is not None
    relations = mem.get_relations("person:sujin")
    assert relations
    assert relations[0].other_entity_id == "project:beta"
    assert relations[0].attrs["role"] == "UI 디자이너"

    mem.close()
