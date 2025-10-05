#!/usr/bin/env python3
from __future__ import annotations

"""OLS regressions of graph connectivity features against alignment score.

Reads the unified CSV dataset and supports per-model, per-year, combined, and
all-model regressions in one invocation. Results are written under
``<outdir>/<model>/<year>/`` with optional ``combined`` and ``all_models``
folders.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

DEFAULT_DATASET = Path("data/aime_regression_dataset.csv")
DEFAULT_MODELS: tuple[str, ...] = (
    "deepseek-r1_32b",
    "gpt-oss_120b",
    "gpt-oss_20b",
    "qwen3_32b",
)
DEFAULT_YEARS: tuple[str, ...] = ("2020", "2021", "2022", "2023", "2024", "2025")

TARGET_COLUMN = "align_score"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return dataset columns that start with ``prefix`` preserving CSV order."""

    return [column for column in df.columns if column.startswith(prefix)]


def _normalise_boolean(series: pd.Series) -> pd.Series:
    def convert(value: object) -> float | pd.NA:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value != 0)
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes"}:
            return 1.0
        if text in {"0", "false", "f", "no"}:
            return 0.0
        return pd.NA

    return series.apply(convert)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    for column in ("model", "id", "year", "exam", "problem"):
        if column in df.columns:
            df[column] = df[column].astype(str)

    if "graph_has_loop" in df.columns:
        df["graph_has_loop"] = _normalise_boolean(df["graph_has_loop"]).astype(float)

    if "align_indices" in df.columns and "align_n_pairs" not in df.columns:
        def count_pairs(value: object) -> float | pd.NA:
            if pd.isna(value):
                return pd.NA
            text = str(value).strip()
            if not text:
                return pd.NA
            try:
                parsed = json.loads(text)
            except Exception:
                return pd.NA
            if isinstance(parsed, list):
                return float(len(parsed))
            return pd.NA

        df["align_n_pairs"] = df["align_indices"].apply(count_pairs)

    numeric_candidates = [
        col
        for col in df.columns
        if col.startswith("graph_") or col.startswith("align_")
    ]
    for col in numeric_candidates:
        if col == "align_indices":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select_rows(
    df: pd.DataFrame,
    models: Iterable[str] | None,
    years: Iterable[str] | None,
) -> pd.DataFrame:
    result = df
    if models is not None:
        result = result[result["model"].isin(list(models))]
    if years is not None:
        result = result[result["year"].isin(list(years))]
    return result.copy()


def run_analysis(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    dest: Path,
    label: str,
) -> None:
    if df.empty:
        print(f"[{label}] no observations; skipping.")
        return

    ensure_dir(dest)

    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        print(f"[{label}] none of the configured graph features are present; skipping.")
        return

    features = df[available_features].apply(pd.to_numeric, errors="coerce").astype(float)
    if TARGET_COLUMN not in df.columns:
        print(f"[{label}] missing target column '{TARGET_COLUMN}'; skipping.")
        return
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(float)

    data = pd.concat([features, target.rename(TARGET_COLUMN)], axis=1).dropna()
    if data.empty:
        print(f"[{label}] all rows dropped after NaN filtering; skipping.")
        return

    X = sm.add_constant(data[available_features], has_constant="add")
    y = data[TARGET_COLUMN]

    if len(y) < 2:
        print(f"[{label}] not enough observations for regression; skipping.")
        return

    model = sm.OLS(y, X).fit()

    summary_path = dest / "ols_summary.txt"
    summary_path.write_text(model.summary().as_text(), encoding="utf-8")

    sg = Stargazer([model])
    sg.title("OLS Regression Results (align_score)")
    try:
        sg.dependent_variable_name("align_score")
    except Exception:
        pass
    html_path = dest / "ols_stargazer.html"
    html_path.write_text(sg.render_html(), encoding="utf-8")
    tex_path = dest / "ols_stargazer.tex"
    tex_path.write_text(sg.render_latex(), encoding="utf-8")

    print(f"[{label}] wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze graph feature influence on alignment score (OLS)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the unified regression dataset CSV",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Model identifiers to analyse (omit for all models in the dataset)",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=list(DEFAULT_YEARS),
        help="Contest years to analyse individually",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/graph_vs_alignment"),
        help="Directory to write analysis artefacts",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip per-model combined regression across selected years",
    )
    parser.add_argument(
        "--skip-overall",
        action="store_true",
        help="Skip the overall regression across all specified models",
    )
    args = parser.parse_args()

    df_full = load_dataset(args.dataset)
    graph_feature_columns = _prefixed_columns(df_full, "graph_")
    requested_models = args.models or sorted(df_full["model"].unique())
    requested_years = [str(year) for year in args.years] if args.years else None

    for model_name in requested_models:
        df_model = select_rows(df_full, [model_name], requested_years)
        if df_model.empty:
            print(f"[{model_name}] no rows found; skipping model.")
            continue

        for year in requested_years or sorted(df_model["year"].unique()):
            df_year = df_model[df_model["year"] == year]
            label = f"{model_name}_{year}"
            dest = args.outdir / model_name / year
            run_analysis(df_year, graph_feature_columns, dest, label)

        if not args.skip_combined:
            df_combined = df_model
            label = f"{model_name}_combined"
            dest = args.outdir / model_name / "combined"
            run_analysis(df_combined, graph_feature_columns, dest, label)

    if not args.skip_overall:
        df_overall = select_rows(df_full, requested_models, requested_years)
        label = "all_models"
        dest = args.outdir / label
        run_analysis(df_overall, graph_feature_columns, dest, label)


if __name__ == "__main__":
    main()
