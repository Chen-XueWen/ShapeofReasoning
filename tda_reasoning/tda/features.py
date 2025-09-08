from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from .filtrations import vr_diagrams
from .persistence import summarize_diagrams


def compute_diagrams(
    points: np.ndarray,
    metric: str = "cosine",
    maxdim: int = 1,
    filtration: str = "vr",
    **kwargs,
) -> Dict[int, np.ndarray]:
    """
    Compute persistence diagrams for a sequence of step embeddings.
    Currently supports 'vr' (Vietoris–Rips) filtration.
    """
    if filtration != "vr":
        raise NotImplementedError(f"Unsupported filtration: {filtration}")
    return vr_diagrams(points, metric=metric, maxdim=maxdim, **kwargs)


# Note: persim-based features (persistence images and landscapes) were removed
# to avoid an extra dependency. If needed in the future, they can be restored.


def betti_curve(diagrams: Dict[int, np.ndarray], resolution: int = 200, dim: int = 0) -> np.ndarray:
    """
    Approximate Betti curve: number of intervals alive at sampled filtration values.
    Returns shape (resolution,).
    """
    dgm = diagrams.get(dim, np.empty((0, 2)))
    if dgm.size == 0:
        return np.zeros((resolution,), dtype=float)
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    finite = np.isfinite(deaths)
    if not finite.any():
        return np.zeros((resolution,), dtype=float)
    t_min = float(np.min(births[finite]))
    t_max = float(np.max(deaths[finite]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        t_min, t_max = 0.0, 1.0
    ts = np.linspace(t_min, t_max, resolution)
    counts = np.zeros_like(ts)
    for b, d in dgm:
        if not np.isfinite(d):
            continue
        counts += (ts >= b) & (ts <= d)
    return counts.astype(float)


def summary_stats(diagrams: Dict[int, np.ndarray]) -> Dict[str, float]:
    return summarize_diagrams(diagrams)


def betti_curve_summary_features(
    diagrams: Dict[int, np.ndarray], resolution: int = 200, dim: int = 0
) -> Dict[str, float]:
    """
    Summarize Betti curve into scalar, interpretable descriptors:
    - betti_peak: max Betti count
    - betti_location: normalized location of the peak in [0, 1]
    - betti_width: normalized FWHM in [0, 1]
    - betti_centroid: normalized first moment in [0, 1]
    - betti_spread: normalized std around centroid in [0, 1]
    - betti_trend: Pearson correlation between t and Betti values in [-1, 1]
    """
    out = {
        f"H{dim}_betti_peak": 0.0,
            f"H{dim}_betti_location": 0.0,
            f"H{dim}_betti_width": 0.0,
            f"H{dim}_betti_centroid": 0.0,
            f"H{dim}_betti_spread": 0.0,
            f"H{dim}_betti_trend": 0.0,
    }

    dgm = diagrams.get(dim, np.empty((0, 2)))
    if dgm.size == 0:
        return out
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    finite = np.isfinite(deaths)
    if not finite.any():
        return out

    t_min = float(np.min(births[finite]))
    t_max = float(np.max(deaths[finite]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        t_min, t_max = 0.0, 1.0
    ts = np.linspace(t_min, t_max, resolution)
    bc = betti_curve(diagrams, resolution=resolution, dim=dim)

    if bc.size != ts.size or not np.isfinite(bc).all():
        return out
    peak = float(np.max(bc)) if bc.size else 0.0
    if peak <= 0.0:
        out["betti_peak"] = 0.0
        # remaining metrics remain 0.0
        return out

    i_peak = int(np.argmax(bc))
    t_range = (t_max - t_min) if (t_max - t_min) > 0 else 1.0
    t_peak_n = float((ts[i_peak] - t_min) / t_range)

    # Centroid and spread treating bc as a density over t
    mass = float(np.trapz(bc, ts))
    if mass > 0:
        centroid = float(np.trapz(bc * ts, ts) / mass)
        var = float(np.trapz(bc * (ts - centroid) ** 2, ts) / mass)
        spread = float(np.sqrt(max(var, 0.0)))
        centroid_n = float((centroid - t_min) / t_range)
        spread_n = float(spread / t_range)
    else:
        centroid_n = 0.0
        spread_n = 0.0

    # FWHM (full width at half maximum)
    half = 0.5 * peak
    mask = bc >= half
    if np.any(mask):
        i0 = int(np.argmax(mask))
        i1 = int(len(bc) - 1 - np.argmax(mask[::-1]))
        width_n = float((ts[i1] - ts[i0]) / t_range) if i1 >= i0 else 0.0
    else:
        width_n = 0.0

    # Trend as Pearson correlation between t and bc
    bc_std = float(bc.std())
    ts_std = float(ts.std())
    if bc_std > 0 and ts_std > 0:
        trend = float(np.corrcoef(ts, bc)[0, 1])
        if not np.isfinite(trend):
            trend = 0.0
    else:
        trend = 0.0

    out.update(
        {
            f"H{dim}_betti_peak": peak,
            f"H{dim}_betti_location": t_peak_n,
            f"H{dim}_betti_width": width_n,
            f"H{dim}_betti_centroid": centroid_n,
            f"H{dim}_betti_spread": spread_n,
            f"H{dim}_betti_trend": trend,
        }
    )
    return out


def assemble_feature_vector(
    diagrams: Dict[int, np.ndarray],
    images: bool = False,
    landscapes: bool = False,
    curves: bool = True,
    stats: bool = True,
    image_dim: int = 1,
    landscape_dim: int = 1,
    curve_dim: int = 0,
) -> Dict[str, np.ndarray | float]:
    """
    Assemble a dict of feature arrays and scalars from diagrams.
    This implementation computes summary statistics and Betti curves only.
    """
    feats: Dict[str, np.ndarray | float] = {}
    if stats:
        feats.update(summary_stats(diagrams))
    # persim-derived features (PI/PL) are disabled by default and not computed
    if curves:
        feats.update(betti_curve_summary_features(diagrams, dim=curve_dim))
        feats.update(betti_curve_summary_features(diagrams, dim=1)) #H1 betti features 
    return feats
