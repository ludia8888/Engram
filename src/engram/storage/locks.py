from __future__ import annotations

import os
from pathlib import Path

from engram.errors import WriterLockError


class WriterLock:
    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise WriterLockError(f"writer lock already held for {self.path}") from exc
        os.write(self._fd, str(os.getpid()).encode("utf-8"))

    def release(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

