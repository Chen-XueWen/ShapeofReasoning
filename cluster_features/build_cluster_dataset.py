#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


@dataclass
class ClusterEvaluation:
    n_clusters: int
    labels: np.ndarray
    silhouette: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster TDA features by correlation and build aggregated dataset.")
    parser.add_argument("--input", default="data/aime_regression_dataset_cleaned.csv", help="Input CSV path.")
    parser.add_argument(
        "--output",
        default="data/aime_regression_dataset_corrcluster.csv",
        help="Output CSV path for clustered feature dataset.",
    )
    parser.add_argument(
        "--membership",
        default="data/aime_tda_feature_clusters.csv",
        help="Optional CSV path to write feature-to-cluster assignments.",
    )
    parser.add_argument(
        "--plot",
        default="plots/tda_feature_corr_silhouette.png",
        help="Path to write silhouette diagnostic plot.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3_32b", "deepseek-r1_32b", "gpt-oss_20b"],
        help="Model identifiers to include when computing correlations.",
    )
    parser.add_argument("--prefix", default="tda_", help="Feature prefix to cluster.")
    parser.add_argument("--min-clusters", type=int, default=2, help="Minimum clusters to evaluate.")
    parser.add_argument("--max-clusters", type=int, default=26, help="Maximum clusters to evaluate.")
    return parser.parse_args()


def normalize_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    subset = df[columns]
    means = subset.mean(axis=0)
    stds = subset.std(axis=0, ddof=0).replace(0, 1)
    return (subset - means) / stds


def drop_constant_columns(df: pd.DataFrame, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    non_constant: list[str] = []
    dropped: list[str] = []
    for column in columns:
        if np.isclose(df[column].std(ddof=0), 0.0):
            dropped.append(column)
        else:
            non_constant.append(column)
    return non_constant, dropped


def correlation_distance(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(method="pearson")
    if corr.isnull().any().any():
        raise ValueError("Correlation matrix contains NaN values; ensure no missing data in feature columns.")
    distance = 1 - corr.abs()
    np.fill_diagonal(distance.values, 0.0)
    return distance


def manual_agglomerative(distance: np.ndarray, n_clusters: int) -> np.ndarray:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("Distance matrix must be square.")
    n_features = distance.shape[0]
    if n_clusters < 1 or n_clusters > n_features:
        raise ValueError("Requested number of clusters is invalid for the feature count.")
    clusters: list[list[int]] = [[idx] for idx in range(n_features)]
    while len(clusters) > n_clusters:
        best_pair: tuple[int, int] | None = None
        best_score = float("inf")
        for i in range(len(clusters) - 1):
            members_i = clusters[i]
            for j in range(i + 1, len(clusters)):
                members_j = clusters[j]
                avg_dist = distance[np.ix_(members_i, members_j)].mean()
                if avg_dist < best_score:
                    best_score = avg_dist
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
    labels = np.empty(n_features, dtype=int)
    for cluster_idx, members in enumerate(clusters):
        for member in members:
            labels[member] = cluster_idx
    return labels


def silhouette_score_distance(distance: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Silhouette score requires at least two clusters.")
    n_samples = distance.shape[0]
    silhouettes: list[float] = []
    for idx in range(n_samples):
        own_label = labels[idx]
        same_mask = labels == own_label
        same_mask[idx] = False
        if same_mask.any():
            intra = distance[idx, same_mask].mean()
        else:
            intra = 0.0
        inter = float("inf")
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = labels == other_label
            if not other_mask.any():
                continue
            inter = min(inter, distance[idx, other_mask].mean())
        if not np.isfinite(inter):
            raise ValueError("Failed to compute silhouette; check cluster assignments.")
        denom = max(intra, inter)
        silhouettes.append(0.0 if denom == 0.0 else (inter - intra) / denom)
    return float(np.mean(silhouettes))


def evaluate_cluster_counts(distance: np.ndarray, candidates: Iterable[int]) -> list[ClusterEvaluation]:
    evaluations: list[ClusterEvaluation] = []
    for n_clusters in candidates:
        if n_clusters <= 1:
            continue
        try:
            labels = manual_agglomerative(distance, n_clusters)
            if np.unique(labels).size < 2:
                continue
            score = silhouette_score_distance(distance, labels)
        except ValueError:
            continue
        evaluations.append(ClusterEvaluation(n_clusters=n_clusters, labels=labels, silhouette=score))
    return evaluations


def pick_best_cluster(evaluations: list[ClusterEvaluation]) -> ClusterEvaluation:
    if not evaluations:
        raise RuntimeError("No valid clustering configuration evaluated; adjust cluster range or check data.")
    evaluations.sort(key=lambda item: (item.silhouette, -item.n_clusters), reverse=True)
    return evaluations[0]


def plot_silhouette(evaluations: list[ClusterEvaluation], path: Path) -> None:
    if not evaluations:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = [item.n_clusters for item in evaluations]
    scores = [item.silhouette for item in evaluations]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(counts, scores, marker="o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score (distance space)")
    ax.set_title("TDA feature correlation clustering diagnostics")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_cluster_columns(df: pd.DataFrame, normalized: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for column, cluster_idx in zip(normalized.columns, labels):
        grouped[int(cluster_idx)].append(column)

    result = df.drop(columns=normalized.columns).copy()
    for cluster_idx in sorted(grouped):
        members = grouped[cluster_idx]
        result[f"tda_corrcluster_{cluster_idx + 1}"] = normalized[members].mean(axis=1)
    return result, grouped


def write_membership(memberships: dict[int, list[str]], path: Path) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cluster_idx, members in sorted(memberships.items()):
        for feature in sorted(members):
            rows.append({"cluster": cluster_idx + 1, "feature": feature})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    membership_path = Path(args.membership)
    plot_path = Path(args.plot)

    df = pd.read_csv(input_path)
    filtered = df[df["model"].isin(args.models)].copy()
    if filtered.empty:
        raise ValueError("No rows match the requested models; check identifiers or source data.")

    feature_columns = [col for col in filtered.columns if col.startswith(args.prefix)]
    if len(feature_columns) < 2:
        raise ValueError(f"Need at least two columns starting with '{args.prefix}' to cluster.")

    feature_columns, dropped = drop_constant_columns(filtered, feature_columns)
    if dropped:
        print(f"Skipping {len(dropped)} constant columns: {', '.join(sorted(dropped))}")
    feature_frame = filtered[feature_columns]
    if feature_frame.isnull().any().any():
        raise ValueError("Feature subset contains missing values; impute or drop rows before clustering.")

    distance_df = correlation_distance(feature_frame)
    distance = distance_df.to_numpy()

    max_clusters = min(args.max_clusters, distance.shape[0])
    min_clusters = min(max(args.min_clusters, 2), max_clusters)
    candidates = range(min_clusters, max_clusters + 1)
    evaluations = evaluate_cluster_counts(distance, candidates)
    plot_silhouette(evaluations, plot_path)

    best = pick_best_cluster(evaluations)
    print(f"Selected {best.n_clusters} clusters (silhouette={best.silhouette:.3f}).")

    normalized = normalize_features(filtered, feature_columns)
    result_df, memberships = build_cluster_columns(filtered, normalized, best.labels)
    if dropped:
        result_df = result_df.drop(columns=dropped, errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Wrote clustered dataset to {output_path}")

    write_membership(memberships, membership_path)
    print(f"Feature-to-cluster assignments written to {membership_path}")
    for cluster_idx in sorted(memberships):
        members = ", ".join(sorted(memberships[cluster_idx]))
        print(f"tda_corrcluster_{cluster_idx + 1}: {members}")


if __name__ == "__main__":
    main()
