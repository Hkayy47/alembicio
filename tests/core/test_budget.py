"""Unit and property tests for the budget tracker and rate pacer (I5, DESIGN.md §7)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from alembicio.config import BudgetConfig
from alembicio.core.budget import BudgetExhaustedError, BudgetTracker, TokenBucket


def _tracker(*, max_usd: float, max_tokens: int) -> BudgetTracker:
    return BudgetTracker(BudgetConfig(max_usd=max_usd, max_tokens=max_tokens))


def test_reserve_within_limit_is_noop() -> None:
    tracker = _tracker(max_usd=10.0, max_tokens=100)
    tracker.reserve(tokens=50, usd=5.0)  # must not raise
    assert tracker.tokens_in == 0  # reserve mutates nothing
    assert tracker.usd_est == 0.0


def test_reserve_over_token_ceiling_raises() -> None:
    tracker = _tracker(max_usd=10.0, max_tokens=100)
    with pytest.raises(BudgetExhaustedError):
        tracker.reserve(tokens=101, usd=0.0)


def test_reserve_over_usd_ceiling_raises() -> None:
    tracker = _tracker(max_usd=1.0, max_tokens=1000)
    with pytest.raises(BudgetExhaustedError):
        tracker.reserve(tokens=1, usd=1.5)


def test_zero_budget_allows_zero_spend() -> None:
    """Local providers configure max_usd: 0; a zero-cost batch must still pass."""
    tracker = _tracker(max_usd=0.0, max_tokens=1000)
    tracker.reserve(tokens=10, usd=0.0)  # must not raise
    tracker.commit(tokens=10, usd=0.0)
    assert tracker.usd_est == 0.0


def test_commit_is_monotone_and_reduces_remaining() -> None:
    tracker = _tracker(max_usd=10.0, max_tokens=100)
    tracker.commit(tokens=30, usd=3.0)
    assert tracker.tokens_in == 30
    assert tracker.remaining_tokens == 70
    assert tracker.remaining_usd == pytest.approx(7.0)
    tracker.commit(tokens=10, usd=1.0)
    assert tracker.tokens_in == 40
    assert tracker.remaining_tokens == 60


@given(
    st.integers(min_value=0, max_value=10_000),
    st.lists(st.integers(min_value=0, max_value=500), max_size=40),
)
def test_reserve_gates_exactly_at_the_ceiling(max_tokens: int, batches: list[int]) -> None:
    """Committed tokens never exceed the ceiling and reserve raises precisely when the
    next batch would breach it."""
    tracker = _tracker(max_usd=1e9, max_tokens=max_tokens)
    for batch in batches:
        would_exceed = tracker.tokens_in + batch > max_tokens
        if would_exceed:
            with pytest.raises(BudgetExhaustedError):
                tracker.reserve(tokens=batch, usd=0.0)
        else:
            tracker.reserve(tokens=batch, usd=0.0)
            tracker.commit(tokens=batch, usd=0.0)
        assert tracker.tokens_in <= max_tokens


@given(st.lists(st.tuples(st.integers(0, 100), st.floats(0, 10)), max_size=50))
def test_ledger_is_never_decreasing(steps: list[tuple[int, float]]) -> None:
    tracker = _tracker(max_usd=1e12, max_tokens=10**12)
    prev_tokens, prev_usd = 0, 0.0
    for tokens, usd in steps:
        tracker.commit(tokens=tokens, usd=usd)
        assert tracker.tokens_in >= prev_tokens
        assert tracker.usd_est >= prev_usd
        prev_tokens, prev_usd = tracker.tokens_in, tracker.usd_est


class _FakeClock:
    """Deterministic monotonic clock the sleeper advances."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_token_bucket_paces_when_empty() -> None:
    clock = _FakeClock()
    slept: list[float] = []

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    bucket = TokenBucket(capacity=100, refill_per_sec=10.0, now=clock)
    bucket.acquire(100, sleeper=sleeper)  # drains the bucket, no wait
    assert slept == []
    bucket.acquire(10, sleeper=sleeper)  # needs one second of refill
    assert sum(slept) == pytest.approx(1.0)


def test_token_bucket_disabled_is_noop() -> None:
    bucket = TokenBucket(capacity=0, refill_per_sec=0.0)

    def boom(_seconds: float) -> None:  # pragma: no cover - must never run
        raise AssertionError("disabled bucket should never sleep")

    bucket.acquire(1_000_000, sleeper=boom)  # must not raise or sleep
