from __future__ import annotations

from engram.retry import RetryPolicy


def test_delay_for_attempt_zero_is_base_delay():
    policy = RetryPolicy(base_delay=1.0, jitter=0.0)
    assert policy.delay_for_attempt(0) == 1.0


def test_delay_for_attempt_exponential_growth():
    policy = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=0.0)
    assert policy.delay_for_attempt(0) == 1.0
    assert policy.delay_for_attempt(1) == 2.0
    assert policy.delay_for_attempt(2) == 4.0
    assert policy.delay_for_attempt(3) == 8.0


def test_delay_for_attempt_capped_at_max():
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=0.0)
    assert policy.delay_for_attempt(10) == 5.0


def test_delay_for_attempt_includes_jitter():
    policy = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=0.5)
    delay = policy.delay_for_attempt(0)
    assert 1.0 <= delay <= 1.5
