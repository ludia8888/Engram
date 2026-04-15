from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from typing import Protocol
from urllib.parse import urlsplit

from .types import Event

_OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
_OPENAI_DEFAULT_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Embedder(Protocol):
    version: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    def __init__(self, *, version: str = "hash-ngram-v1", dim: int = 256):
        self.version = version
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text, self.dim) for text in texts]


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _OPENAI_DEFAULT_MODEL,
        dimensions: int | None = None,
        base_url: str | None = None,
        semantic_space_id: str | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self._requested_dimensions = dimensions
        self.base_url = base_url
        self.semantic_space_id = semantic_space_id or _default_openai_space_id(base_url)
        self.dim = dimensions or _OPENAI_DEFAULT_DIMS.get(model, 0)
        dims_label = str(dimensions) if dimensions is not None else "default"
        self.version = f"openai:{self.semantic_space_id}:{model}:{dims_label}:v1"
        self._client = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client_instance().embeddings.create(
            model=self.model,
            input=texts,
            **({"dimensions": self._requested_dimensions} if self._requested_dimensions is not None else {}),
        )
        vectors = [list(item.embedding) for item in response.data]
        if len(vectors) != len(texts):
            raise ValueError(f"OpenAI embedder returned {len(vectors)} embeddings for {len(texts)} texts")
        if not vectors:
            return []

        dim = len(vectors[0])
        if any(len(vector) != dim for vector in vectors):
            raise ValueError("OpenAI embedder returned inconsistent embedding dimensions")
        if self._requested_dimensions is not None and dim != self._requested_dimensions:
            raise ValueError(
                f"OpenAI embedder returned dim {dim}, expected requested dim {self._requested_dimensions}"
            )
        self.dim = dim
        return vectors

    def _client_instance(self):
        if self._client is None:
            client_class = _load_openai_client_class()
            self._client = client_class(api_key=self.api_key, base_url=self.base_url)
        return self._client


def event_semantic_text(event: Event) -> str:
    data = json.dumps(event.data, ensure_ascii=False, sort_keys=True)
    return f"{event.type} {data} {event.source_role}"


def embedding_to_blob(values: list[float], *, dim: int) -> bytes:
    if len(values) != dim:
        raise ValueError(f"expected embedding dim {dim}, got {len(values)}")
    return struct.pack(f"<{dim}f", *values)


def embedding_from_blob(blob: bytes, *, dim: int) -> list[float]:
    expected_size = struct.calcsize(f"<{dim}f")
    if len(blob) != expected_size:
        raise ValueError(f"expected blob size {expected_size}, got {len(blob)}")
    return list(struct.unpack(f"<{dim}f", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _hash_embed(text: str, dim: int) -> list[float]:
    normalized = text.strip().lower()
    if not normalized:
        return [0.0] * dim

    padded = f"  {normalized}  "
    vector = [0.0] * dim
    for index in range(max(len(padded) - 2, 1)):
        ngram = padded[index : index + 3]
        digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0] * dim
    return [value / norm for value in vector]


def _load_openai_client_class():
    try:
        module = importlib.import_module("openai")
    except ImportError as exc:
        raise RuntimeError(
            'OpenAIEmbedder requires the optional "openai" dependency. Install it with '
            '`pip install "engram[openai]"`.'
        ) from exc

    client_class = getattr(module, "OpenAI", None)
    if client_class is None:
        raise RuntimeError('Installed "openai" package does not expose OpenAI client class.')
    return client_class


def _default_openai_space_id(base_url: str | None) -> str:
    if base_url is None:
        return "api.openai.com"

    parsed = urlsplit(base_url)
    if parsed.netloc:
        path = parsed.path.rstrip("/")
        return f"{parsed.netloc.lower()}{path}" if path else parsed.netloc.lower()

    return base_url.rstrip("/").lower()
