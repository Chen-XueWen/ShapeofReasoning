from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AlignmentResult:
    indices: List[Tuple[int, int]]  # (model_step_idx, gold_step_idx)
    score: float  # average similarity over aligned pairs
    coverage: float  # fraction of gold steps aligned


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(np.clip(an @ bn, -1.0, 1.0))


def align_steps(
    model_emb: np.ndarray,
    gold_emb: np.ndarray,
    gap_penalty: float = 0.0,
) -> AlignmentResult:
    """
    Monotonic alignment (DP) maximizing total cosine similarity with optional gap penalty.
    Returns aligned index pairs with average similarity and gold coverage.
    """
    n, m = model_emb.shape[0], gold_emb.shape[0]
    S = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            S[i, j] = cosine_sim(model_emb[i], gold_emb[j])

    # DP tables
    dp = np.full((n + 1, m + 1), -1e9, dtype=float)
    bt = np.zeros((n + 1, m + 1, 2), dtype=int)
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and j < m:
                v = dp[i, j] + S[i, j]
                if v > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = v
                    bt[i + 1, j + 1] = (i, j)
            if i < n:
                v = dp[i, j] - gap_penalty
                if v > dp[i + 1, j]:
                    dp[i + 1, j] = v
                    bt[i + 1, j] = (i, j)
            if j < m:
                v = dp[i, j] - gap_penalty
                if v > dp[i, j + 1]:
                    dp[i, j + 1] = v
                    bt[i, j + 1] = (i, j)

    # Backtrack to get path ending at (n, m)
    i, j = n, m
    pairs: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        pi, pj = bt[i, j]
        if pi == i - 1 and pj == j - 1:
            pairs.append((i - 1, j - 1))
        i, j = pi, pj
    pairs.reverse()

    # Compute average similarity over aligned pairs and coverage
    sim_sum = 0.0
    for (i, j) in pairs:
        sim_sum += S[i, j]
    avg_sim = sim_sum / max(1, len(pairs))
    gold_covered = len(set(j for _, j in pairs)) / max(1, m)
    return AlignmentResult(indices=pairs, score=avg_sim, coverage=gold_covered)

