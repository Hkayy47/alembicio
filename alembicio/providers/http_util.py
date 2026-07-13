"""Shared HTTP retry/circuit-breaker helpers for embedding providers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import httpx

from alembicio.providers.base import ProviderPausedError, TransientProviderError

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_CONSECUTIVE_FAILURES = 5
MAX_ATTEMPTS = 6


def jittered_backoff(attempt: int, *, base: float = 0.5, cap: float = 30.0) -> float:
    """Return exponential backoff with full jitter in ``[0, min(cap, base * 2**attempt)]``."""
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0.0, ceiling)


class RetryState:
    """Track consecutive transient failures for a circuit breaker."""

    def __init__(self) -> None:
        self.consecutive_failures = 0

    def record_success(self) -> None:
        """Reset the failure streak after a successful call."""
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Increment the failure streak and trip the breaker at the threshold."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise ProviderPausedError(
                f"provider circuit open after {self.consecutive_failures} consecutive failures"
            )


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retry_state: RetryState,
    sleeper: Callable[[float], None] = time.sleep,
    **kwargs: object,
) -> httpx.Response:
    """Perform an HTTP request with jittered exponential backoff on 429/5xx.

    Args:
        client: The httpx client to use.
        method: HTTP method.
        url: Request URL.
        retry_state: Circuit-breaker state shared across calls.
        sleeper: Sleep function, injectable for tests.
        **kwargs: Passed through to ``client.request``.

    Returns:
        The successful response.

    Raises:
        TransientProviderError: On retryable HTTP failures before the breaker trips.
        ProviderPausedError: After ``MAX_CONSECUTIVE_FAILURES`` consecutive failures.
        httpx.HTTPError: On non-retryable transport errors after retries exhaust.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.request(method, url, **kwargs)  # type: ignore[arg-type]
        except httpx.HTTPError as exc:
            last_error = exc
            retry_state.record_failure()
            if attempt + 1 >= MAX_ATTEMPTS:
                raise TransientProviderError(str(exc)) from exc
            sleeper(jittered_backoff(attempt))
            continue

        if response.status_code in RETRYABLE_STATUS:
            last_error = TransientProviderError(
                f"HTTP {response.status_code} from {url}"
            )
            retry_state.record_failure()
            if attempt + 1 >= MAX_ATTEMPTS:
                raise last_error
            sleeper(jittered_backoff(attempt))
            continue

        if response.status_code >= 400:
            response.raise_for_status()

        retry_state.record_success()
        return response

    msg = f"request failed after {MAX_ATTEMPTS} attempts"
    raise TransientProviderError(msg) from last_error


def conservative_token_estimate(texts: list[str]) -> int:
    """Return a conservative chars/3.5 token estimate."""
    total_chars = sum(len(text) for text in texts)
    return max(1, int(total_chars / 3.5 + 0.999))
