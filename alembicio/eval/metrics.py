"""Retrieval quality metrics (NumPy-only)."""

from __future__ import annotations


def recall_at_k(ranked: list[str], expected: list[str], *, k: int) -> float:
    """Compute known-item recall@k."""
    raise NotImplementedError("recall_at_k")


def mrr(ranked: list[str], expected: list[str]) -> float:
    """Compute mean reciprocal rank for the first expected hit."""
    raise NotImplementedError("mrr")


def overlap_at_k(old_ranked: list[str], new_ranked: list[str], *, k: int) -> float:
    """Compute overlap@k between two ranked lists."""
    raise NotImplementedError("overlap_at_k")
