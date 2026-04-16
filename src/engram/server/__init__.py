from __future__ import annotations

import atexit
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from engram.canonical import Extractor
from engram.engram import Engram
from engram.errors import EngramError, QueueFullError, ValidationError, WriterLockError
from engram.semantic import Embedder

from .routes import router

_cleanup_lock = threading.Lock()


def create_app(
    user_id: str = "default",
    path: str | None = None,
    session_id: str | None = None,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
    auto_flush: bool = True,
) -> FastAPI:
    config: dict[str, Any] = {
        "user_id": user_id,
        "path": path,
        "session_id": session_id,
        "extractor": extractor,
        "embedder": embedder,
        "auto_flush": auto_flush,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        closed = False

        def cleanup():
            nonlocal closed
            with _cleanup_lock:
                if closed:
                    return
                closed = True
            app.state.engram.close()

        engram = Engram(
            user_id=config["user_id"],
            path=config["path"],
            session_id=config["session_id"],
            extractor=config["extractor"],
            embedder=config["embedder"],
            auto_flush=config["auto_flush"],
        )
        app.state.engram = engram
        app.state.config = config
        atexit.register(cleanup)
        try:
            yield
        finally:
            cleanup()
            atexit.unregister(cleanup)

    app = FastAPI(title="Engram", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, exc: QueueFullError):
        return JSONResponse(status_code=503, content={"detail": "Queue full, try again later"})

    @app.exception_handler(WriterLockError)
    async def writer_lock_handler(request: Request, exc: WriterLockError):
        return JSONResponse(status_code=503, content={"detail": "Writer lock held by another process"})

    @app.exception_handler(EngramError)
    async def engram_error_handler(request: Request, exc: EngramError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
