from __future__ import annotations

import types

import pytest

import engram.openai_meaning_analyzer as meaning_module
import engram.semantic as semantic_module
from engram.errors import ValidationError
from engram.openai_meaning_analyzer import OpenAIMeaningAnalyzer
from engram.types import Event

from tests.conftest import dt


def _event() -> Event:
    return Event(
        id="event-1",
        seq=1,
        observed_at=dt("2026-05-01T10:00:00Z"),
        effective_at_start=None,
        effective_at_end=None,
        recorded_at=dt("2026-05-01T10:00:01Z"),
        type="entity.create",
        data={"id": "user:alice", "type": "user", "attrs": {"label": "Busan-1499", "role": "traveler"}},
        extraction_run_id=None,
        source_turn_id=None,
        source_role="manual",
        confidence=0.91,
        reason="user label mentions Busan-1499 traveler",
        time_confidence="unknown",
        caused_by=None,
        schema_version=1,
    )


def _install_fake_openai(monkeypatch, responses, *, requests=None, prompts=None):
    class FakeCompletions:
        def create(self, **kwargs):
            if requests is not None:
                requests.append(kwargs)
            if prompts is not None:
                prompts.append(kwargs["messages"][1]["content"])
            if not responses:
                raise AssertionError("no fake OpenAI responses left")
            payload = responses.pop(0)
            if isinstance(payload, dict) and "_raw_response" in payload:
                return payload["_raw_response"]
            if isinstance(payload, str):
                content = payload
            else:
                import json

                content = json.dumps(payload, ensure_ascii=False)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content, refusal=None),
                        finish_reason="stop",
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(meaning_module, "_load_openai_client_class", lambda: FakeOpenAI)


def test_openai_meaning_analyzer_parses_event_units_and_query_plan(monkeypatch):
    requests: list[dict] = []
    _install_fake_openai(
        monkeypatch,
        [
            {
                "units": [
                    {"kind": "protected_phrase", "value": "Busan-1499", "confidence": 0.98},
                    {"kind": "facet", "key": "role", "value": "traveler", "confidence": 0.82},
                ]
            },
            {
                "units": [
                    {"kind": "protected_phrase", "value": "Busan-1499", "confidence": 0.97},
                    {"kind": "canonical_key", "value": "label:busan-1499", "confidence": 0.93},
                ],
                "planner_confidence": 0.91,
            },
        ],
        requests=requests,
    )

    analyzer = OpenAIMeaningAnalyzer(api_key="test-key", base_url="https://example.test/v1")
    event_analysis = analyzer.analyze_event(_event())
    query_plan = analyzer.plan_query("Busan-1499 traveler")

    assert analyzer.version == "openai-meaning-analyzer:example.test/v1:gpt-5.4-mini:v1"
    assert event_analysis.units[0].kind == "protected_phrase"
    assert event_analysis.units[0].normalized_value == "busan-1499"
    assert event_analysis.units[1].kind == "facet"
    assert event_analysis.units[1].key == "role"
    assert query_plan.units[0].kind == "protected_phrase"
    assert query_plan.units[1].kind == "canonical_key"
    assert query_plan.fallback_terms == ["busan-1499", "busan", "1499", "traveler"]
    assert query_plan.planner_confidence == 0.91
    assert requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("response_payload", "message"),
    [
        ("{not-json", "malformed JSON"),
        ({"units": [{"kind": "unknown", "value": "Busan"}]}, "must be one of"),
        ({"units": [{"kind": "facet", "value": "traveler"}]}, "key must be a non-empty string"),
        ({"units": [{"kind": "protected_phrase", "value": "Busan", "confidence": 2.0}]}, "between 0.0 and 1.0"),
    ],
)
def test_openai_meaning_analyzer_rejects_invalid_model_output(monkeypatch, response_payload, message):
    _install_fake_openai(monkeypatch, [response_payload])
    analyzer = OpenAIMeaningAnalyzer()

    with pytest.raises(ValidationError, match=message):
        analyzer.analyze_event(_event())


def test_openai_meaning_analyzer_raises_clear_error_when_sdk_missing(monkeypatch):
    def fake_import_module(name: str):
        if name == "openai":
            raise ImportError("missing openai")
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(semantic_module.importlib, "import_module", fake_import_module)
    analyzer = OpenAIMeaningAnalyzer()

    with pytest.raises(RuntimeError, match='pip install "engram\\[openai\\]"'):
        analyzer.plan_query("Busan-1499")


def test_openai_meaning_analyzer_surfaces_refusal_reason(monkeypatch):
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
    analyzer = OpenAIMeaningAnalyzer()

    with pytest.raises(ValidationError, match="safety refusal"):
        analyzer.plan_query("Busan-1499")


def test_openai_meaning_analyzer_surfaces_truncated_json_error(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        [
            {
                "_raw_response": types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content='{"units":[', refusal=None),
                            finish_reason="length",
                        )
                    ]
                )
            }
        ],
    )
    analyzer = OpenAIMeaningAnalyzer()

    with pytest.raises(ValidationError, match="truncated"):
        analyzer.analyze_event(_event())
