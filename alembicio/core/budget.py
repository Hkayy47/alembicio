"""Token and USD budget enforcement plus rate-limit pacing (DESIGN.md §7, I5).

Two collaborators live here:

* :class:`BudgetTracker` -- the monotone spend ledger with *pre-reservation*. It is
  the correctness object behind I5: reservation is checked **before** any provider
  spend, and committed spend only ever increases. A batch whose embed step is
  abandoned (crash, budget ceiling, provider error) simply never calls ``commit``,
  so no partial spend is recorded.
* :class:`TokenBucket` -- a throughput pacer (``tpm``/``rpm``). Pacing is a
  performance concern, not a correctness invariant, and is a no-op when the
  configured limit is zero (local, unthrottled providers).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from alembicio.config import BudgetConfig


class BudgetExhaustedError(Exception):
    """Raised when a batch would exceed a configured token or USD budget ceiling."""


class BudgetTracker:
    """Monotone token/USD counters with pre-spend reservation (I5).

    The tracker holds the *committed* spend. ``reserve`` answers the question "may I
    spend this batch?" without mutating anything, so an unanswered reservation (the
    embed never completes) leaves no trace. ``commit`` records durable spend after the
    upsert that consumed it and only ever moves the counters upward.
    """

    def __init__(self, limits: BudgetConfig) -> None:
        """Initialize a tracker against configured ceilings.

        Args:
            limits: The ``max_usd`` / ``max_tokens`` ceilings from config.
        """
        self._limits = limits
        self.tokens_in: int = 0
        self.usd_est: float = 0.0

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining before the ceiling (never negative)."""
        return max(0, self._limits.max_tokens - self.tokens_in)

    @property
    def remaining_usd(self) -> float:
        """USD remaining before the ceiling (never negative)."""
        return max(0.0, self._limits.max_usd - self.usd_est)

    def reserve(self, *, tokens: int, usd: float) -> None:
        """Check that a batch fits under both ceilings; raise if it does not.

        This mutates nothing: it is the in-memory pre-reservation of DESIGN.md §4/§7.
        A caller that proceeds past ``reserve`` must call :meth:`commit` only after the
        spend is durably reflected downstream.

        Args:
            tokens: Estimated input tokens for the batch.
            usd: Estimated USD cost for the batch.

        Returns:
            None.

        Raises:
            BudgetExhaustedError: If committing this batch would breach a ceiling.
        """
        if self.tokens_in + tokens > self._limits.max_tokens:
            msg = (
                f"token budget exhausted: {self.tokens_in} + {tokens} "
                f"> {self._limits.max_tokens}"
            )
            raise BudgetExhaustedError(msg)
        if self.usd_est + usd > self._limits.max_usd:
            msg = (
                f"usd budget exhausted: {self.usd_est} + {usd} "
                f"> {self._limits.max_usd}"
            )
            raise BudgetExhaustedError(msg)

    def commit(self, *, tokens: int, usd: float) -> None:
        """Record durable spend for a batch; counters only ever increase.

        Args:
            tokens: Input tokens actually spent by the batch.
            usd: USD cost actually spent by the batch.

        Returns:
            None.
        """
        self.tokens_in += tokens
        self.usd_est += usd


class TokenBucket:
    """Classic token-bucket pacer used for ``tpm``/``rpm`` throttling (DESIGN.md §7).

    A ``capacity`` of ``0`` disables the bucket entirely (unthrottled local providers).
    The clock and sleeper are injectable so the pacer is deterministic under test.
    """

    def __init__(
        self,
        *,
        capacity: int,
        refill_per_sec: float,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a bucket.

        Args:
            capacity: Maximum tokens the bucket holds; ``0`` disables throttling.
            refill_per_sec: Steady-state refill rate in tokens per second.
            now: Monotonic clock source, injectable for tests.
        """
        self._capacity = capacity
        self._refill_per_sec = refill_per_sec
        self._now = now
        self._tokens: float = float(capacity)
        self._last = now()

    def _refill(self) -> None:
        """Add tokens accrued since the last observation, capped at capacity."""
        current = self._now()
        elapsed = max(0.0, current - self._last)
        self._last = current
        self._tokens = min(
            float(self._capacity), self._tokens + elapsed * self._refill_per_sec
        )

    def acquire(
        self, tokens: int, *, sleeper: Callable[[float], None] = time.sleep
    ) -> None:
        """Block until ``tokens`` are available, then consume them.

        Args:
            tokens: Tokens the caller wishes to consume.
            sleeper: Sleep function, injectable for tests.

        Returns:
            None.
        """
        if self._capacity <= 0 or self._refill_per_sec <= 0:
            return
        need = float(min(tokens, self._capacity))
        while True:
            self._refill()
            if self._tokens >= need:
                self._tokens -= need
                return
            deficit = need - self._tokens
            sleeper(deficit / self._refill_per_sec)
