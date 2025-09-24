#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

INPUT_PATH = Path("data/aime_regression_dataset_cleaned.csv")
OUTPUT_PATH = Path("data/aime_regression_dataset_kmeans.csv")
N_CLUSTERS = 6
RANDOM_STATE = 0
FEATURE_PREFIX = "tda_"


def normalize_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return z-scored subset; guard against zero variance columns."""
    subset = df[columns]
    means = subset.mean(axis=0)
    stds = subset.std(axis=0, ddof=0).replace(0, 1)
    return (subset - means) / stds


def cluster_feature_columns(normalized: pd.DataFrame) -> dict[str, int]:
    """Cluster columns using k-means and return column -> cluster index mapping."""
    model = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=RANDOM_STATE)
    labels = model.fit_predict(normalized.T)  # Treat features as samples.
    return {column: int(cluster_idx) for column, cluster_idx in zip(normalized.columns, labels)}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    feature_columns = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    if len(feature_columns) < N_CLUSTERS:
        raise ValueError(
            f"Found {len(feature_columns)} feature columns with prefix '{FEATURE_PREFIX}',"
            f" but need at least {N_CLUSTERS} to form that many clusters."
        )

    normalized = normalize_features(df, feature_columns)
    column_to_cluster = cluster_feature_columns(normalized)

    cluster_members: dict[int, list[str]] = defaultdict(list)
    for column, cluster_idx in column_to_cluster.items():
        cluster_members[cluster_idx].append(column)

    result = df.drop(columns=feature_columns).copy()
    for cluster_idx in range(N_CLUSTERS):
        members = cluster_members.get(cluster_idx)
        if not members:
            raise RuntimeError(f"Cluster {cluster_idx} is empty; try a different random seed.")
        result[f"tda_cluster_{cluster_idx + 1}"] = normalized[members].mean(axis=1)

    for cluster_idx in range(N_CLUSTERS):
        members = ", ".join(sorted(cluster_members[cluster_idx]))
        print(f"tda_cluster_{cluster_idx + 1}: {members}")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
