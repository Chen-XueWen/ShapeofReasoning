#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/aime_regression_dataset_cleaned.csv")
OUTPUT_PATH = Path("data/aime_regression_dataset_qualitative.csv")

GROUPS = {
    "tda_H0_features": [
        "tda_H0_count",
        "tda_H0_total_life",
        "tda_H0_max_life",
        "tda_H0_mean_life",
        "tda_H0_entropy",
        "tda_H0_skewness",
    ],
    "tda_H1_features": [
        "tda_H1_count",
        "tda_H1_total_life",
        "tda_H1_max_life",
        "tda_H1_mean_life",
        "tda_H1_entropy",
        "tda_H1_skewness",
        "tda_H1_max_birth",
        "tda_H1_max_death",
    ],
    "tda_H0_betti": [
        "tda_H0_betti_centroid",
        "tda_H0_betti_spread",
        "tda_H0_betti_width",
    ],
    "tda_H1_betti": [
        "tda_H1_betti_peak",
        "tda_H1_betti_location",
        "tda_H1_betti_width",
        "tda_H1_betti_centroid",
        "tda_H1_betti_spread",
    ],
    "tda_H0_landscape": [
        "tda_H0_landscape_mean",
        "tda_H0_landscape_max",
        "tda_H0_landscape_area",
    ],
    "tda_H1_landscape": [
        "tda_H1_landscape_mean",
        "tda_H1_landscape_max",
        "tda_H1_landscape_area",
    ],
}


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    grouped_columns = sorted({col for cols in GROUPS.values() for col in cols})
    means = df[grouped_columns].mean(axis=0)
    stds = df[grouped_columns].std(axis=0, ddof=0).replace(0, 1)
    normalized = df[grouped_columns].subtract(means).divide(stds)

    result = df.drop(columns=grouped_columns).copy()
    for group_name, columns in GROUPS.items():
        result[group_name] = normalized[columns].mean(axis=1)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
