"""Local fastembed provider (demo extra, D12)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from alembicio.providers.base import RateLimits, TokenEstimate
from alembicio.providers.constants import resolve_limits
from alembicio.providers.http_util import conservative_token_estimate

_FASTEMBED: Any | None = None
_TEXT_EMBEDDING: Any | None = None


def _load_fastembed() -> tuple[Any, Any]:
    """Import fastembed lazily so core installs without the demo extra."""
    global _FASTEMBED, _TEXT_EMBEDDING
    if _FASTEMBED is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            msg = "fastembed is not installed; install with `uv sync --extra demo`"
            raise ImportError(msg) from exc
        _FASTEMBED = TextEmbedding
        _TEXT_EMBEDDING = TextEmbedding
    return _FASTEMBED, _TEXT_EMBEDDING


class FastEmbedLocalProvider:
    """Keyless local embeddings via fastembed."""

    def __init__(self, *, pricing_overrides: dict[str, float] | None = None) -> None:
        """Initialize the provider."""
        self._pricing_overrides = pricing_overrides
        self._models: dict[str, Any] = {}

    def _model(self, name: str) -> Any:
        if name not in self._models:
            _, text_embedding = _load_fastembed()
            self._models[name] = text_embedding(model_name=name)
        return self._models[name]

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dims: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a batch locally."""
        if not texts:
            return np.zeros((0, dims or 0), dtype=np.float32)
        embedder = self._model(model)
        vectors = list(embedder.embed(list(texts)))
        matrix = np.array(vectors, dtype=np.float32)
        if dims is not None and matrix.shape[1] > dims:
            matrix = matrix[:, :dims]
        return matrix

    def estimate_tokens(self, texts: Sequence[str], *, model: str) -> TokenEstimate:
        """Return a conservative local token estimate."""
        _ = model
        return TokenEstimate(tokens=conservative_token_estimate(list(texts)), estimated=True)

    def count_tokens(self, texts: Sequence[str], *, model: str) -> int:
        """Return the conservative token estimate for a batch."""
        return self.estimate_tokens(texts, model=model)["tokens"]

    def limits(self, *, model: str) -> RateLimits:
        """Return local (unthrottled) limits."""
        return resolve_limits(model, overrides=self._pricing_overrides)
