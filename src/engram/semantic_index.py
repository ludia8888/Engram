from __future__ import annotations

from .search_terms import event_search_terms
from .semantic import Embedder, embedding_to_blob, event_semantic_text
from .time_utils import to_rfc3339, utcnow


class SemanticIndexer:
    def __init__(self, store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def index_missing(self) -> int:
        embedding_events = self.store.events_missing_embeddings(self.embedder.version)
        term_events = self.store.events_missing_search_terms()

        if not embedding_events and not term_events:
            return 0

        embedding_rows: list[tuple[str, str, int, bytes, str]] = []
        if embedding_events:
            texts = [event_semantic_text(event) for event in embedding_events]
            embeddings = self.embedder.embed_texts(texts)
            if len(embeddings) != len(embedding_events):
                raise ValueError(
                    f"embedder returned {len(embeddings)} embeddings for {len(embedding_events)} events"
                )
            if embeddings:
                embedding_dim = len(embeddings[0])
                if any(len(embedding) != embedding_dim for embedding in embeddings):
                    raise ValueError("embedder returned inconsistent embedding dimensions")
            else:
                embedding_dim = self.embedder.dim

            indexed_at = to_rfc3339(utcnow())
            for event, embedding in zip(embedding_events, embeddings, strict=True):
                embedding_rows.append(
                    (
                        event.id,
                        self.embedder.version,
                        embedding_dim,
                        embedding_to_blob(embedding, dim=embedding_dim),
                        indexed_at,
                    )
                )

        with self.store.transaction() as tx:
            self.store.append_event_embeddings(tx, embedding_rows)
            for event in term_events:
                self.store.append_event_search_terms(tx, event.id, event_search_terms(event))
        return len(embedding_rows) + len(term_events)
