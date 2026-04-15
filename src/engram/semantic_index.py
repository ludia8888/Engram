from __future__ import annotations

from .semantic import Embedder, embedding_to_blob, event_semantic_text
from .time_utils import to_rfc3339, utcnow


class SemanticIndexer:
    def __init__(self, store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def index_missing(self) -> int:
        events = self.store.events_missing_embeddings(self.embedder.version)
        if not events:
            return 0

        texts = [event_semantic_text(event) for event in events]
        embeddings = self.embedder.embed_texts(texts)
        if len(embeddings) != len(events):
            raise ValueError(
                f"embedder returned {len(embeddings)} embeddings for {len(events)} events"
            )
        if embeddings:
            embedding_dim = len(embeddings[0])
            if any(len(embedding) != embedding_dim for embedding in embeddings):
                raise ValueError("embedder returned inconsistent embedding dimensions")
        else:
            embedding_dim = self.embedder.dim

        indexed_at = to_rfc3339(utcnow())
        rows: list[tuple[str, str, int, bytes, str]] = []
        for event, embedding in zip(events, embeddings, strict=True):
            rows.append(
                (
                    event.id,
                    self.embedder.version,
                    embedding_dim,
                    embedding_to_blob(embedding, dim=embedding_dim),
                    indexed_at,
                )
            )

        with self.store.transaction() as tx:
            self.store.append_event_embeddings(tx, rows)
        return len(rows)
