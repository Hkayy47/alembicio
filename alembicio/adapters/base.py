"""Store adapter protocol and shared types."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
import numpy.typing as npt


class DocRecord(TypedDict):
    """Document row claimed for embedding."""

    doc_id: str
    text: str | None
    content_ref: str | None
    content_hash: str


class VectorRecord(TypedDict):
    """Embedded or projected vector ready for upsert."""

    doc_id: str
    content_hash: str
    vector: npt.NDArray[np.float32]
    provenance: Literal["embedded", "projected"]


class StoreInfo(TypedDict):
    """Summary returned by inspect()."""

    dims: int
    row_count: int
    pending_count: int
    disk_estimate_bytes: int


class ReconcileReport(TypedDict):
    """Summary returned by reconcile()."""

    tombstones_applied: int
    dirty_requeued: int
    orphans_removed: int


class ModelSpec(TypedDict):
    """Target model metadata passed to prepare()."""

    provider: str
    model: str
    dim: int


class StoreAdapter(Protocol):
    """Backend-neutral vector store interface."""

    def inspect(self) -> StoreInfo:
        """Return dims, counts, text availability, and disk estimate."""
        ...

    def prepare(self, target: ModelSpec) -> None:
        """Add target column/collection and install change capture."""
        ...

    def pending_count(self) -> int:
        """Count rows not backfill-complete per pending predicate."""
        ...

    def claim_batch(self, *, limit: int) -> list[DocRecord]:
        """Claim a batch of pending documents for embedding."""
        ...

    def upsert_vectors(self, batch: list[VectorRecord]) -> None:
        """Idempotently write vectors keyed by (doc_id, content_hash)."""
        ...

    def reconcile(self) -> ReconcileReport:
        """Apply deletes-win sweep and dirty re-enqueue."""
        ...

    def build_index(self, *, concurrently: bool = True) -> None:
        """Build ANN index on the target embedding column."""
        ...

    def search(
        self,
        vector: npt.NDArray[np.float32],
        *,
        space: Literal["old", "new"],
        k: int,
    ) -> list[str]:
        """Search old or new space and return doc ids."""
        ...

    def flip_read_path(
        self,
        *,
        active: Literal["old", "new"],
        canary_pct: int = 0,
    ) -> None:
        """Atomically flip the read path."""
        ...

    def decommission(self) -> None:
        """Drop migration artifacts from the store."""
        ...
