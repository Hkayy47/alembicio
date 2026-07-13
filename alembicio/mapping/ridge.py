"""Ridge-regularized affine mapping between embedding spaces."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from alembicio.mapping.procrustes import MappingArtifact


def fit_ridge(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    lam: float | str = "auto",
) -> MappingArtifact:
    """Fit a ridge-regularized map from source to target anchors."""
    raise NotImplementedError("fit_ridge")
