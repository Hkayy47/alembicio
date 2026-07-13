"""Embedding provider protocol, error taxonomy, and shared types (DESIGN.md §7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict

import numpy as np
import numpy.typing as npt


class RateLimits(TypedDict):
    """Provider rate and pricing limits."""

    tpm: int
    rpm: int
    max_input_tokens: int
    usd_per_mtok: float


class TokenEstimate(TypedDict):
    """Token count with an estimation flag."""

    tokens: int
    estimated: bool


class ProviderError(Exception):
    """Base class for embedding-provider failures the worker knows how to handle."""


class ProviderPausedError(ProviderError):
    """Circuit breaker open after consecutive transient failures (DESIGN.md §7)."""


class TransientProviderError(ProviderError):
    """A retryable provider failure (HTTP 429/5xx, timeouts).

    The worker backs off and retries these; a run of consecutive transient failures
    trips the circuit breaker and pauses the migration as ``PAUSED_ERROR`` (I5/I6).
    """


class PoisonInputError(ProviderError):
    """A permanent, input-specific failure (over token limit, empty, provider 4xx).

    The offending items are dead-lettered with :attr:`reason` and the run continues;
    they are excluded from subsequent claims so backfill can still reach completion.
    """

    def __init__(self, bad_indices: Sequence[int], *, reason: str) -> None:
        """Initialize a poison error.

        Args:
            bad_indices: Positions within the embedded batch that are poison.
            reason: Dead-letter reason code (e.g. ``token_limit``, ``empty``).
        """
        self.bad_indices: tuple[int, ...] = tuple(bad_indices)
        self.reason = reason
        super().__init__(f"poison input at indices {self.bad_indices}: {reason}")


class EmbeddingProvider(Protocol):
    """Backend-neutral embedding API."""

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dims: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a batch of texts.

        Args:
            texts: The batch of input strings.
            model: Target model identifier.
            dims: Optional output dimensionality (MRL truncation).

        Returns:
            A ``(len(texts), d)`` float32 array of embeddings.
        """
        ...

    def count_tokens(self, texts: Sequence[str], *, model: str) -> int:
        """Return the input token count for a batch.

        Args:
            texts: The batch of input strings.
            model: Target model identifier.

        Returns:
            Total input tokens (exact when a tokenizer is available, else estimated).
        """
        ...

    def limits(self, *, model: str) -> RateLimits:
        """Return rate limits and pricing for a model.

        Args:
            model: Target model identifier.

        Returns:
            The provider's rate/pricing limits for the model.
        """
        ...
