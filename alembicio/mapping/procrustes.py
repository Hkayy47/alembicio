"""Orthogonal Procrustes mapping between embedding spaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MappingArtifact:
    """Fitted linear map between source and target spaces."""

    matrix: npt.NDArray[np.float64]
    source_model: str
    target_model: str
    dims: tuple[int, int]
    n_anchors: int
    recovery: float

    def apply(self, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply the mapping to source-space vectors."""
        raise NotImplementedError("MappingArtifact.apply")

    def save(self, path: Path) -> None:
        """Persist artifact to disk."""
        raise NotImplementedError("MappingArtifact.save")

    @classmethod
    def load(cls, path: Path) -> MappingArtifact:
        """Load artifact from disk."""
        raise NotImplementedError("MappingArtifact.load")


def fit_procrustes(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    allow_rectangular: bool = True,
) -> MappingArtifact:
    """Fit an orthogonal Procrustes map from source to target anchors."""
    raise NotImplementedError("fit_procrustes")
