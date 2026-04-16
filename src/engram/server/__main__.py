from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from engram.config import build_embedder, build_extractor, build_meaning_analyzer
    from engram.server import create_app

    app = create_app(
        user_id=os.environ.get("ENGRAM_USER_ID", "default"),
        path=os.environ.get("ENGRAM_PATH"),
        session_id=os.environ.get("ENGRAM_SESSION_ID"),
        extractor=build_extractor(),
        embedder=build_embedder(),
        meaning_analyzer=build_meaning_analyzer(),
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
