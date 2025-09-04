import numpy as np


def summarize_diagrams(dgms: dict[int, np.ndarray]) -> dict[str, float]:
    """
    Simple summary statistics over persistence diagrams per dimension.
    Returns a flat dict like {"H0_count": ..., "H1_total_life": ..., ...}.
    """
    feats: dict[str, float] = {}
    for dim, dgm in dgms.items():
        if dgm.size == 0:
            feats[f"H{dim}_count"] = 0.0
            feats[f"H{dim}_total_life"] = 0.0
            feats[f"H{dim}_max_life"] = 0.0
            feats[f"H{dim}_entropy"] = 0.0
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        life = np.where(np.isfinite(deaths), deaths - births, 0.0)
        count = float(dgm.shape[0])
        total_life = float(np.sum(life))
        max_life = float(np.max(life) if life.size else 0.0)

        # Persistent entropy (natural log): H = -sum_i p_i log p_i where p_i = life_i / sum(life)
        if total_life > 0.0:
            positive = life > 1e-12
            p = (life[positive] / total_life).astype(float)
            entropy = float(-np.sum(p * np.log(p))) if p.size else 0.0
        else:
            entropy = 0.0

        feats[f"H{dim}_count"] = count
        feats[f"H{dim}_total_life"] = total_life
        feats[f"H{dim}_max_life"] = max_life
        feats[f"H{dim}_entropy"] = entropy
    return feats
