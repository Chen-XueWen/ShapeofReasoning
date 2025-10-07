#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Variance Inflation Factors for TDA features per model."
    )
    parser.add_argument(
        "--input",
        default="data/aime_regression_dataset_cleaned.csv",
        help="Input CSV containing AIME regression dataset.",
    )
    parser.add_argument(
        "--output",
        default="cluster_features/tda_vif_by_model.csv",
        help="Where to write the aggregated VIF table (CSV).",
    )
    parser.add_argument(
        "--prefix",
        default="tda_",
        help="Feature prefix to include in the VIF calculation.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum rows required per model after cleaning to compute VIF.",
    )
    return parser.parse_args()


def select_feature_columns(columns: Iterable[str], prefix: str) -> list[str]:
    return [column for column in columns if column.startswith(prefix)]


def compute_vif(feature_df: pd.DataFrame) -> list[tuple[str, float]]:
    exog = add_constant(feature_df.values, has_constant="add")
    vifs: list[tuple[str, float]] = []
    for idx, column in enumerate(feature_df.columns, start=1):
        vif_value = variance_inflation_factor(exog, idx)
        vifs.append((column, float(vif_value)))
    return vifs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    feature_columns = select_feature_columns(df.columns, args.prefix)

    records: list[dict[str, object]] = []

    for model, model_df in df.groupby("model", sort=True):
        vif_pairs = compute_vif(model_df[feature_columns])
        for feature, vif_value in vif_pairs:
            records.append({"model": model, "feature": feature, "vif": vif_value})

    result_df = pd.DataFrame.from_records(records).sort_values(["model", "vif"], ascending=[True, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Wrote VIF table for {result_df['model'].nunique()} models and {len(result_df)} rows to {output_path}.")

if __name__ == "__main__":
    main()
