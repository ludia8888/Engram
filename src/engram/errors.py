class EngramError(Exception):
    """Base error for Engram."""


class ValidationError(EngramError):
    """Raised when public API input is invalid."""


class QueueFullError(EngramError):
    """Raised when a turn cannot be enqueued within the configured timeout."""


class WriterLockError(EngramError):
    """Raised when another writer already owns the same storage root."""

