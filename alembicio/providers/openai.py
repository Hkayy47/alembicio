"""OpenAI embedding provider (DESIGN.md §7)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx
import numpy as np
import numpy.typing as npt

from alembicio.providers.base import (
    PoisonInputError,
    RateLimits,
    TokenEstimate,
)
from alembicio.providers.constants import resolve_limits
from alembicio.providers.http_util import (
    RetryState,
    conservative_token_estimate,
    request_with_retries,
)

_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


class OpenAIProvider:
    """OpenAI embeddings API client with retries and a circuit breaker."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        pricing_overrides: dict[str, float] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: API key; defaults to ``OPENAI_API_KEY``.
            pricing_overrides: Optional per-model USD/M-token overrides from yaml.
            client: Optional httpx client for tests.
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            msg = "OPENAI_API_KEY is not set"
            raise ValueError(msg)
        self._api_key = key
        self._pricing_overrides = pricing_overrides
        self._client = client or httpx.Client(timeout=60.0)
        self._retry_state = RetryState()

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dims: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a batch of texts via the OpenAI embeddings API."""
        if not texts:
            return np.zeros((0, dims or 0), dtype=np.float32)
        payload: dict[str, object] = {"input": list(texts), "model": model}
        if dims is not None:
            payload["dimensions"] = dims
        response = request_with_retries(
            self._client,
            "POST",
            _OPENAI_EMBED_URL,
            retry_state=self._retry_state,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code == 400:
            raise PoisonInputError(list(range(len(texts))), reason="provider_4xx")
        data = response.json()
        rows = sorted(data["data"], key=lambda item: item["index"])
        vectors = np.array([row["embedding"] for row in rows], dtype=np.float32)
        return vectors

    def estimate_tokens(self, texts: Sequence[str], *, model: str) -> TokenEstimate:
        """Return token count; OpenAI has no public count endpoint, so estimate."""
        _ = model
        return TokenEstimate(tokens=conservative_token_estimate(list(texts)), estimated=True)

    def count_tokens(self, texts: Sequence[str], *, model: str) -> int:
        """Return the conservative token estimate for a batch."""
        return self.estimate_tokens(texts, model=model)["tokens"]

    def limits(self, *, model: str) -> RateLimits:
        """Return rate limits and pricing for a model."""
        return resolve_limits(model, overrides=self._pricing_overrides)
