from __future__ import annotations

import types

import pytest

import engram.semantic as semantic_module
from engram.errors import ValidationError
from engram.openai_extractor import OpenAIExtractor
from engram.types import QueueItem, RawTurn

from tests.conftest import dt, install_fake_openai as _install_fake_openai


def _recent_turns() -> list[RawTurn]:
    return [
        RawTurn(
            id="prior-1",
            session_id="sess-1",
            observed_at=dt("2026-05-01T09:00:00Z"),
            user="나는 채식주의자야",
            assistant="알겠어, 기억해둘게.",
            metadata={},
        )
    ]


def _queue_item() -> QueueItem:
    return QueueItem(
        turn_id="turn-1",
        session_id="sess-1",
        observed_at=dt("2026-05-02T10:00:00Z"),
        user="지난주에 부산으로 이사했고 Bob이 내 매니저야",
        assistant="알겠어, 기억해둘게.",
        metadata={},
    )


def test_openai_extractor_parses_and_normalizes_events(monkeypatch):
    requests: list[dict] = []
    _install_fake_openai(
        monkeypatch,
        [
            {
                "events": [
                    {
                        "type": "entity.create",
                        "data": {"id": "self", "type": "user", "attrs": {"name": "Alice"}},
                        "confidence": 0.91,
                        "reason": "사용자가 자기 자신을 Alice라고 표현함",
                    },
                    {
                        "type": "entity.update",
                        "data": {"id": "self", "attrs": {"location": "Busan"}},
                        "confidence": 0.94,
                        "reason": "사용자가 현재 거주지를 부산이라고 명시함",
                    },
                    {
                        "type": "entity.update",
                        "data": {"id": "self", "attrs": {"diet": "vegetarian"}},
                        "confidence": 0.81,
                        "reason": "사용자가 채식주의자라고 말함",
                    },
                    {
                        "type": "relation.create",
                        "data": {
                            "source": "self",
                            "target": "bob",
                            "type": "manager",
                            "attrs": {"scope": "work"},
                        },
                        "confidence": 0.88,
                        "reason": "사용자가 Bob이 자신의 매니저라고 명시함",
                    },
                ]
            }
        ],
        requests=requests,
    )

    extractor = OpenAIExtractor(api_key="test-key", base_url="https://example.test/v1")
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: _recent_turns()[:limit],
    )

    events = extractor.extract(_queue_item())

    assert len(events) == 2
    assert events[0].type == "entity.create"
    assert events[0].data == {
        "id": "user:alice",
        "type": "user",
        "attrs": {"name": "Alice", "location": "Busan", "diet": "vegetarian"},
    }
    assert events[1].type == "relation.create"
    assert events[1].data == {
        "source": "user:alice",
        "target": "entity:bob",
        "type": "manager",
        "attrs": {"scope": "work"},
    }
    assert requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("response_payload", "message"),
    [
        ("{not-json", "malformed JSON"),
        ({"events": [{"type": "memory.note", "data": {}, "confidence": 0.5, "reason": "bad"}]}, "unsupported"),
        ({"events": [{"type": "entity.update", "data": {"id": "self", "attrs": {}}, "confidence": 2.0, "reason": "bad"}]}, "between 0.0 and 1.0"),
        ({"events": [{"type": "entity.update", "data": {"id": "self"}, "confidence": 0.5, "reason": "missing attrs"}]}, "attrs must be an object"),
    ],
)
def test_openai_extractor_rejects_invalid_model_output(monkeypatch, response_payload, message):
    _install_fake_openai(monkeypatch, [response_payload])
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    with pytest.raises((ValidationError, RuntimeError), match=message):
        extractor.extract(_queue_item())


def test_openai_extractor_raises_clear_error_when_sdk_missing(monkeypatch):
    def fake_import_module(name: str):
        if name == "openai":
            raise ImportError("missing openai")
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(semantic_module.importlib, "import_module", fake_import_module)
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    with pytest.raises(RuntimeError, match='pip install "engram\\[openai\\]"'):
        extractor.extract(_queue_item())


def test_openai_extractor_requires_runtime_binding(monkeypatch):
    _install_fake_openai(monkeypatch, [{"events": []}])
    extractor = OpenAIExtractor()

    with pytest.raises(RuntimeError, match="must be bound"):
        extractor.extract(_queue_item())


def test_openai_extractor_surfaces_refusal_reason(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            {
                "_raw_response": types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content=None, refusal="safety refusal"),
                            finish_reason="stop",
                        )
                    ]
                )
            }
        ],
    )
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    with pytest.raises(ValidationError, match="safety refusal"):
        extractor.extract(_queue_item())


def test_openai_extractor_surfaces_truncated_json_error(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            {
                "_raw_response": types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content='{"events":[', refusal=None),
                            finish_reason="length",
                        )
                    ]
                )
            }
        ],
    )
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    with pytest.raises(ValidationError, match="truncated"):
        extractor.extract(_queue_item())


def test_openai_extractor_handles_hangul_jamo_slug(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            {
                "events": [
                    {
                        "type": "entity.create",
                        "data": {"id": "ㄱㅣㅁ 철수", "type": "person", "attrs": {"name": "김철수"}},
                        "confidence": 0.9,
                        "reason": "사용자가 이름을 명시함",
                    }
                ]
            }
        ],
    )
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    events = extractor.extract(_queue_item())

    assert events[0].data["id"] == "person:ㄱㅣㅁ-철수"


def test_normalize_batch_merges_create_and_subsequent_update(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            {
                "events": [
                    {
                        "type": "entity.create",
                        "data": {"id": "self", "type": "user", "attrs": {"name": "Alice"}},
                        "confidence": 0.9,
                        "reason": "create",
                    },
                    {
                        "type": "entity.update",
                        "data": {"id": "self", "attrs": {"location": "Busan"}},
                        "confidence": 0.95,
                        "reason": "update after create",
                    },
                ]
            }
        ],
    )
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    events = extractor.extract(_queue_item())

    assert len(events) == 1
    assert events[0].type == "entity.create"
    assert events[0].data["attrs"] == {"name": "Alice", "location": "Busan"}
    assert events[0].confidence == 0.95


def test_normalize_batch_does_not_merge_create_and_delete(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            {
                "events": [
                    {
                        "type": "entity.create",
                        "data": {"id": "self", "type": "user", "attrs": {"name": "Alice"}},
                        "confidence": 0.9,
                        "reason": "create",
                    },
                    {
                        "type": "entity.delete",
                        "data": {"id": "self"},
                        "confidence": 0.8,
                        "reason": "delete after create",
                    },
                ]
            }
        ],
    )
    extractor = OpenAIExtractor()
    extractor.bind_runtime_context(
        safe_user_id="alice",
        recent_turns_provider=lambda item, limit: [],
    )

    events = extractor.extract(_queue_item())

    assert len(events) == 2
    assert events[0].type == "entity.create"
    assert events[1].type == "entity.delete"
