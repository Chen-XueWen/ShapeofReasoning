from typing import Literal, Optional
from ripser import ripser
import numpy as np
from scipy.stats import skew
from persim import PersLandscapeExact

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
                f"{dim}_max_birth": 0.0,
                f"{dim}_max_death": 0.0,
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
            f"{dim}_max_birth": float(np.max(births) if births.size else 0.0),
            f"{dim}_max_death": float(np.max(deaths[finite]) if np.any(finite) else 0.0),
        })
    return feats

def landscape_features(
    diagrams: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute persistence landscape features from Vietoris–Rips persistent homology diagrams.

    - For each dimension, compute `num_landscapes` landscapes, each with `num_levels` levels.
    - Summarise each landscape level with mean, max, and total area.
    """
    

    feats: dict[str, float] = {}
    dgms = [d for d in diagrams.values()]
    for dim, dgm in diagrams.items():
        if dgm.size == 0:
            feats.update({
                f"{dim}_landscape_mean": 0.0,
                f"{dim}_landscape_max": 0.0,
                f"{dim}_landscape_area": 0.0,
            })
            continue
        dim = int(dim[-1])
        pl = PersLandscapeExact(dgms=dgms, hom_deg=dim)
        landscape = pl.critical_pairs
        xs = np.array([x for x, _ in landscape[0]])
        ys = np.array([y for _, y in landscape[0]])
        full_xs = [np.array([x for x, _ in level]) for level in landscape]
        full_ys = [np.array([y for _, y in level]) for level in landscape]
        mean_val = float(np.mean(ys)) if ys.size > 0 else 0.0
        max_val = float(np.max(ys)) if ys.size > 0 else 0.0
        area_val = float(np.trapezoid(ys, xs)) if ys.size > 1 else 0.0
        total_area_val = float(sum(np.trapezoid(y, x) for x, y in zip(full_xs, full_ys) if y.size > 1)) if ys.size > 1 else 0.0
        feats.update({
            f"H{dim}_landscape_mean": mean_val,
            f"H{dim}_landscape_max": max_val,
            f"H{dim}_landscape_area": area_val,
        })
    return feats
