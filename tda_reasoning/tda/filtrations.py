from __future__ import annotations

from typing import Dict, Optional
from ripser import ripser
import numpy as np


def _pairwise_distances(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if metric == "euclidean":
        # (x - y)^2 = x^2 + y^2 - 2x·y
        sq = (X ** 2).sum(axis=1, keepdims=True)
        D2 = sq + sq.T - 2 * (X @ X.T)
        D2 = np.maximum(D2, 0.0)
        return np.sqrt(D2)
    elif metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
        D = 1.0 - np.clip(S, -1.0, 1.0)
        return D
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def vr_diagrams(
    X: np.ndarray,
    metric: str = "cosine",
    maxdim: int = 1,
    thresh: Optional[float] = None,
    return_dgm_dict: bool = True,
) -> Dict[int, np.ndarray] | dict:
    """
    Compute Vietoris–Rips persistent homology from a point cloud or distance matrix.

    - If metric is not 'euclidean', we compute a pairwise distance matrix first.
    - Uses ripser if available; otherwise raises a helpful error.
    """

    if metric == "euclidean":
        out = ripser(X, maxdim=maxdim, thresh=thresh)
    else:
        D = _pairwise_distances(X, metric=metric)
        # ripser distance_matrix path expects a real threshold; default to inf if None
        t = float("inf") if thresh is None else float(thresh)
        out = ripser(D, maxdim=maxdim, thresh=t, distance_matrix=True)

    dgms = out.get("dgms", [])
    if return_dgm_dict:
        return {i: np.asarray(d) for i, d in enumerate(dgms)}
    return out
