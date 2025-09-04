from typing import Literal

import numpy as np
from tda_reasoning.tda.betti import betti_curve_summary_features

from .filtrations import vr_diagrams
from .persistence import summarize_diagrams


def compute_diagrams(
    points: np.ndarray,
    metric: Literal["euclidean", "cosine"] = "cosine",
    maxdim: int = 1,
    filtration: Literal["vr"] = "vr",
    **kwargs,
) -> dict[int, np.ndarray]:
    """
    Compute persistence diagrams for a sequence of step embeddings.
    Currently supports 'vr' (Vietoris–Rips) filtration.
    """
    if filtration != "vr":
        raise NotImplementedError(f"Unsupported filtration: {filtration}")
    return vr_diagrams(points, metric=metric, maxdim=maxdim, **kwargs)


# Note: persim-based features (persistence images and landscapes) were removed
# to avoid an extra dependency. If needed in the future, they can be restored.
def summary_stats(diagrams: dict[int, np.ndarray]) -> dict[str, float]:
    return summarize_diagrams(diagrams)

def assemble_feature_vector(
    diagrams: dict[int, np.ndarray],
    images: bool = False,
    landscapes: bool = False,
    curves: bool = True,
    stats: bool = True,
    image_dim: int = 1,
    landscape_dim: int = 1,
    curve_dim: int = 0,
) -> dict[str, np.ndarray | float]:
    """
    Assemble a dict of feature arrays and scalars from diagrams.
    This implementation computes summary statistics and Betti curves only.
    """
    feats: dict[str, np.ndarray | float] = {}
    if stats:
        feats.update(summary_stats(diagrams))
    # persim-derived features (PI/PL) are disabled by default and not computed
    if curves:
        feats.update(betti_curve_summary_features(diagrams, dim=curve_dim))
    return feats
