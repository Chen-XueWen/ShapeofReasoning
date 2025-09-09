from typing import Literal, Optional
from ripser import ripser
import numpy as np
from scipy.stats import skew


def vr_diagrams(
    X: np.ndarray,
    metric: Literal["euclidean", "cosine"] = "cosine",
    maxdim: int = 1,
    thresh: Optional[float] = None,
) -> dict[int, np.ndarray]:
    """
    Compute Vietoris–Rips persistent homology from a point cloud or distance matrix.

    - If metric is not 'euclidean', we compute a pairwise distance matrix first.
    - Uses ripser if available; otherwise raises a helpful error.
    """
    out = ripser(X, maxdim=maxdim, metric=metric, thresh=thresh if thresh is not None else float("inf"))

    return {i: dgms for i, dgms in enumerate(out["dgms"])}

def vr_features(
    diagrams: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Summarise features of Vietoris–Rips persistent homology up to dimension `maxdim`.
    """
    feats: dict[str, float] = {}
    for dim, dgm in diagrams.items():
        if dgm.size == 0:
            feats.update({
                f"{dim}_count": 0.0, 
                f"{dim}_total_life": 0.0,
                f"{dim}_max_life": 0.0,
                f"{dim}_mean_life": 0.0,
                f"{dim}_entropy": 0.0,
                f"{dim}_skewness": 0.0,
            })
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        finite = np.isfinite(deaths)
        life = np.where(np.isfinite(deaths), deaths - births, 0.0)
        total_life = float(np.sum(life))
        max_life = float(np.max(life) if life.size else 0.0)
        finite = np.isfinite(deaths)
        if total_life > 0.0:
            positive = life > 0.0
            p = (life[positive] / total_life).astype(float)
            entropy = float(-np.sum(p * np.log(p))) if p.size else 0.0
        else:
            entropy = 0.0
        skewness = float(skew(life)) if life.size > 2 else 0.0
        feats.update({
            f"{dim}_count": float(dgm.shape[0]), 
            f"{dim}_total_life": total_life,
            f"{dim}_max_life": max_life,
            f"{dim}_mean_life": total_life / float(finite.shape[0]) if finite.shape[0] > 0 else 0.0,
            f"{dim}_entropy": entropy,
            f"{dim}_skewness": skewness,
        })
    return feats
