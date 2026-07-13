"""Post-backfill reconciliation sweeps."""

from __future__ import annotations

from alembicio.adapters.base import ReconcileReport


def reconcile_pending() -> ReconcileReport:
    """Run deletes-win and dirty re-enqueue reconciliation."""
    raise NotImplementedError("reconcile_pending")
