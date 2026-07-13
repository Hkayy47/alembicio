"""Golden query set loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class GoldenQuery(TypedDict):
    """Single known-item golden query."""

    q: str
    expect: list[str]
    k: int


def load_golden(path: Path) -> list[GoldenQuery]:
    """Load golden queries from jsonl."""
    raise NotImplementedError("load_golden")
