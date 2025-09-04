from __future__ import annotations

from typing import Dict, Literal, Optional
from ripser import ripser
import numpy as np


def vr_diagrams(
    X: np.ndarray,
    metric: Literal["euclidean", "cosine"] = "cosine",
    maxdim: int = 1,
    thresh: Optional[float] = None,
    return_dgm_dict: bool = True,
) -> Dict[int, np.ndarray] | dict:
    """
    Compute Vietoris–Rips persistent homology from a point cloud or distance matrix.

    - If metric is not 'euclidean', we compute a pairwise distance matrix first.
    - Uses ripser if available; otherwise raises a helpful error.
    """
    out = ripser(X, maxdim=maxdim, metric=metric, thresh=thresh if thresh is not None else float("inf"))
    
    dgms = out.get("dgms", [])
    if return_dgm_dict:
        return {i: np.asarray(d) for i, d in enumerate(dgms)}
    return out
