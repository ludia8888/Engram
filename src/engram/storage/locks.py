from __future__ import annotations

import json
import os
from pathlib import Path

from engram.errors import WriterLockError
from engram.time_utils import to_rfc3339, utcnow


class WriterLock:
    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        for _attempt in range(2):
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self._write_metadata(self._fd)
                return
            except FileExistsError as exc:
                if not self._recover_stale_lock():
                    raise WriterLockError(f"writer lock already held for {self.path}") from exc
        raise WriterLockError(f"writer lock already held for {self.path}")

    def release(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _write_metadata(self, fd: int) -> None:
        payload = {
            "pid": os.getpid(),
            "created_at": to_rfc3339(utcnow()),
        }
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)

    def _recover_stale_lock(self) -> bool:
        metadata = self._read_metadata()
        pid = metadata.get("pid") if metadata else None
        if not isinstance(pid, int):
            return False
        if self._pid_is_alive(pid):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        return True

    def _read_metadata(self) -> dict | None:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                return {"pid": int(raw)}
            except ValueError:
                return None
        if isinstance(parsed, int):
            return {"pid": parsed}
        if isinstance(parsed, dict):
            return parsed
        return None

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
