import numpy as np

def betti_curve(dgm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate Betti curve: number of intervals alive at sampled filtration values.
    Returns shape (resolution,).
    """
    if dgm.size == 0:
        return np.array([]), np.array([])
    ts = []
    tracker = []
    for birth, death in dgm:
        tracker.append((birth, +1))
        tracker.append((death, -1))
    tracker.sort(key=lambda x: x[0])
    counts = []
    count = 0
    for t, delta in tracker:
        if not ts or t != ts[-1]:
            if np.isfinite(t):
                ts.append(t)
                counts.append(count)
        count += delta
    return np.array(ts), np.array(counts, dtype=float)


def betti_features(
    diagrams: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Summarise Betti curve into scalar, interpretable descriptors:
    - betti_peak: max Betti count
    - betti_location: normalized location of the peak in [0, 1]
    - betti_width: normalized FWHM in [0, 1]
    - betti_centroid: normalized first moment in [0, 1]
    - betti_spread: normalized std around centroid in [0, 1]
    - betti_trend: Pearson correlation between t and Betti values in [-1, 1]
    """
    out = {
        "H0_betti_peak": 0.0,
        "H0_betti_location": 0.0,
        "H0_betti_width": 0.0,
        "H0_betti_centroid": 0.0,
        "H0_betti_spread": 0.0,
        "H0_betti_trend": 0.0,
        "H1_betti_peak": 0.0,
        "H1_betti_location": 0.0,
        "H1_betti_width": 0.0,
        "H1_betti_centroid": 0.0,
        "H1_betti_spread": 0.0,
        "H1_betti_trend": 0.0,
    }
    for dim, dgm in diagrams.items():
        if dgm.size == 0:
            continue
        deaths = dgm[:, 1]
        finite = np.isfinite(deaths)
        if not finite.any():
            continue
        
        ts, bc = betti_curve(dgm)
        t_min, t_max = np.min(ts), np.max(ts)

        if bc.size != ts.size or not np.isfinite(bc).all():
            continue
        peak = float(np.max(bc)) if bc.size else 0.0
        if peak <= 0.0:
            out[f"H{dim}_betti_peak"] = 0.0
            # remaining metrics remain 0.0
            continue

        i_peak = int(np.argmax(bc))
        t_range = t_max - t_min
        t_peak_n = float((ts[i_peak] - t_min) / t_range)

        # Centroid and spread treating bc as a density over t
        mass = float(np.trapezoid(bc, ts))
        if mass > 0:
            centroid = float(np.trapezoid(bc * ts, ts) / mass)
            var = float(np.trapezoid(bc * (ts - centroid) ** 2, ts) / mass)
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

        out[f"{dim}_betti_peak"] = peak
        out[f"{dim}_betti_location"] = t_peak_n
        out[f"{dim}_betti_width"] = width_n
        out[f"{dim}_betti_centroid"] = centroid_n
        out[f"{dim}_betti_spread"] = spread_n
        out[f"{dim}_betti_trend"] = trend
    return out