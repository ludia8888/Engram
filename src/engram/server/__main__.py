from __future__ import annotations

import os

from engram.canonical import NullExtractor
from engram.meaning_index import NullMeaningAnalyzer
from engram.semantic import HashEmbedder


def _build_extractor():
    name = os.environ.get("ENGRAM_EXTRACTOR", "null")
    if name == "null":
        return NullExtractor()
    if name == "openai":
        from engram.openai_extractor import OpenAIExtractor

        return OpenAIExtractor(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown extractor: {name}")


def _build_embedder():
    name = os.environ.get("ENGRAM_EMBEDDER", "hash")
    if name == "hash":
        return HashEmbedder()
    if name == "openai":
        from engram.semantic import OpenAIEmbedder

        dims = os.environ.get("ENGRAM_OPENAI_EMBED_DIMS")
        return OpenAIEmbedder(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            dimensions=int(dims) if dims else None,
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown embedder: {name}")


def _build_meaning_analyzer():
    name = os.environ.get("ENGRAM_MEANING_ANALYZER", "null")
    if name == "null":
        return NullMeaningAnalyzer()
    if name == "openai":
        from engram.openai_meaning_analyzer import OpenAIMeaningAnalyzer

        return OpenAIMeaningAnalyzer(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_MEANING_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown meaning analyzer: {name}")


def main() -> None:
    import uvicorn

    from engram.server import create_app

    app = create_app(
        user_id=os.environ.get("ENGRAM_USER_ID", "default"),
        path=os.environ.get("ENGRAM_PATH"),
        session_id=os.environ.get("ENGRAM_SESSION_ID"),
        extractor=_build_extractor(),
        embedder=_build_embedder(),
        meaning_analyzer=_build_meaning_analyzer(),
        auto_flush=os.environ.get("ENGRAM_AUTO_FLUSH", "true").lower() in ("true", "1", "yes"),
    )
    uvicorn.run(
        app,
        host=os.environ.get("ENGRAM_HOST", "127.0.0.1"),
        port=int(os.environ.get("ENGRAM_PORT", "8000")),
        log_level=os.environ.get("ENGRAM_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
