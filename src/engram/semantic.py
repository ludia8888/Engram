from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Protocol

from .types import Event


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
