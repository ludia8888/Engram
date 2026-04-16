from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.5

    def delay_for_attempt(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter_amount = delay * self.jitter * random.random()
        return delay + jitter_amount


@dataclass(slots=True)
class RetryState:
    turn_id: str
    attempt: int = 0
    last_error: str | None = None
    next_eligible_at: float | None = None

    def schedule_next(self, policy: RetryPolicy) -> None:
        self.next_eligible_at = time.monotonic() + policy.delay_for_attempt(self.attempt)
