"""Default provider rate limits and pricing (DESIGN.md §7, D12)."""

from __future__ import annotations

from alembicio.providers.base import RateLimits

DEFAULT_LIMITS: dict[str, RateLimits] = {
    "text-embedding-ada-002": {
        "tpm": 1_000_000,
        "rpm": 3_000,
        "max_input_tokens": 8191,
        "usd_per_mtok": 0.10,
    },
    "text-embedding-3-small": {
        "tpm": 5_000_000,
        "rpm": 5_000,
        "max_input_tokens": 8191,
        "usd_per_mtok": 0.02,
    },
    "text-embedding-3-large": {
        "tpm": 5_000_000,
        "rpm": 5_000,
        "max_input_tokens": 8191,
        "usd_per_mtok": 0.13,
    },
    "gemini-embedding-001": {
        "tpm": 1_000_000,
        "rpm": 1_500,
        "max_input_tokens": 2048,
        "usd_per_mtok": 0.15,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "tpm": 0,
        "rpm": 0,
        "max_input_tokens": 512,
        "usd_per_mtok": 0.0,
    },
    "BAAI/bge-small-en-v1.5": {
        "tpm": 0,
        "rpm": 0,
        "max_input_tokens": 512,
        "usd_per_mtok": 0.0,
    },
}


def resolve_limits(
    model: str,
    *,
    overrides: dict[str, float] | None = None,
) -> RateLimits:
    """Return rate limits for a model, applying yaml pricing overrides when present."""
    base = DEFAULT_LIMITS.get(
        model,
        {"tpm": 0, "rpm": 0, "max_input_tokens": 8192, "usd_per_mtok": 0.0},
    )
    limits: RateLimits = {
        "tpm": base["tpm"],
        "rpm": base["rpm"],
        "max_input_tokens": base["max_input_tokens"],
        "usd_per_mtok": base["usd_per_mtok"],
    }
    if overrides and model in overrides:
        limits["usd_per_mtok"] = overrides[model]
    return limits
