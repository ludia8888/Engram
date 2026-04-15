from __future__ import annotations

import json
import types
from datetime import UTC, datetime

import engram.openai_extractor as _extractor_module


def dt(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(UTC)


def install_fake_openai(
    monkeypatch,
    responses: list[dict | str],
    *,
    requests: list[dict] | None = None,
    prompts: list[str] | None = None,
) -> None:
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
            content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
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

    monkeypatch.setattr(_extractor_module, "_load_openai_client_class", lambda: FakeOpenAI)

