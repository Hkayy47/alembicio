"""Verification report rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    *,
    path_md: Path,
    path_json: Path,
    metrics: dict[str, Any],
    gate_passed: bool,
    projected_row_count: int,
    synthetic: bool,
) -> None:
    """Write report.md and report.json verification artifacts."""
    raise NotImplementedError("write_report")
