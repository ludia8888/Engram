from __future__ import annotations

import types

import pytest

import engram.semantic as semantic_module
from engram.semantic import OpenAIEmbedder


def test_openai_embedder_returns_vectors_and_updates_dim(monkeypatch):
    class FakeEmbeddings:
        def create(self, *, model, input, dimensions=None):
            assert model == "text-embedding-3-small"
            assert dimensions is None
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
                    types.SimpleNamespace(embedding=[0.0, 1.0, 0.0]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            assert api_key == "test-key"
            assert base_url == "https://example.test/v1"
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(semantic_module, "_load_openai_client_class", lambda: FakeOpenAI)

    embedder = OpenAIEmbedder(
        api_key="test-key",
        model="text-embedding-3-small",
        base_url="https://example.test/v1",
    )

    vectors = embedder.embed_texts(["alpha", "beta"])

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert embedder.dim == 3


def test_openai_embedder_version_changes_with_model_and_dimensions():
    default = OpenAIEmbedder(model="text-embedding-3-small")
    resized = OpenAIEmbedder(model="text-embedding-3-small", dimensions=256)
    different_model = OpenAIEmbedder(model="text-embedding-3-large")
    different_backend = OpenAIEmbedder(model="text-embedding-3-small", base_url="https://proxy.example/v1")
    overridden_space = OpenAIEmbedder(model="text-embedding-3-small", semantic_space_id="tenant-a")

    assert default.version != resized.version
    assert default.version != different_model.version
    assert resized.version != different_model.version
    assert default.version != different_backend.version
    assert default.version != overridden_space.version


def test_openai_embedder_validates_inconsistent_embedding_dimensions(monkeypatch):
    class FakeEmbeddings:
        def create(self, *, model, input, dimensions=None):
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
                    types.SimpleNamespace(embedding=[0.0, 1.0]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(semantic_module, "_load_openai_client_class", lambda: FakeOpenAI)

    embedder = OpenAIEmbedder()

    with pytest.raises(ValueError, match="inconsistent embedding dimensions"):
        embedder.embed_texts(["alpha", "beta"])


def test_openai_embedder_validates_requested_dimension(monkeypatch):
    class FakeEmbeddings:
        def create(self, *, model, input, dimensions=None):
            assert dimensions == 5
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(semantic_module, "_load_openai_client_class", lambda: FakeOpenAI)

    embedder = OpenAIEmbedder(dimensions=5)

    with pytest.raises(ValueError, match="expected requested dim 5"):
        embedder.embed_texts(["alpha"])


def test_openai_embedder_raises_clear_error_when_sdk_missing(monkeypatch):
    def fake_import_module(name: str):
        if name == "openai":
            raise ImportError("missing openai")
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(semantic_module.importlib, "import_module", fake_import_module)

    embedder = OpenAIEmbedder()

    with pytest.raises(RuntimeError, match='pip install "engram\\[openai\\]"'):
        embedder.embed_texts(["hello"])


def test_openai_embedder_validates_embedding_count(monkeypatch):
    class FakeEmbeddings:
        def create(self, *, model, input, dimensions=None):
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1.0, 0.0, 0.0])])

    class FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(semantic_module, "_load_openai_client_class", lambda: FakeOpenAI)

    embedder = OpenAIEmbedder()

    with pytest.raises(ValueError, match="returned 1 embeddings for 2 texts"):
        embedder.embed_texts(["alpha", "beta"])
