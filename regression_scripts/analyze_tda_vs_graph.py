#!/usr/bin/env python3
from __future__ import annotations

"""OLS regressions of TDA features against graph properties.

The script operates on the unified CSV dataset and can run regressions per
model, per year, and across all models. Output structure mirrors the target
variable first: ``<outdir>/<target>/<model>/<year>/`` plus ``combined`` and
``all_models`` folders when enabled.
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return dataset columns that start with ``prefix`` preserving CSV order."""

    return [column for column in df.columns if column.startswith(prefix)]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    for column in ("model", "id", "year", "exam", "problem"):
        if column in df.columns:
            df[column] = df[column].astype(str)

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
        if col.startswith("tda_") or col.startswith("graph_")
    ]
    for col in numeric_candidates:
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


def run_regression(
    df: pd.DataFrame,
    dependent: str,
    feature_columns: Sequence[str],
    dest: Path,
    label: str,
) -> None:
    if dependent not in df.columns:
        print(f"[{label}] column '{dependent}' missing; skipping.")
        return

    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        print(f"[{label}] no TDA feature columns present; skipping.")
        return

    ensure_dir(dest)

    features = df[available_features].apply(pd.to_numeric, errors="coerce").astype(float)
    target = pd.to_numeric(df[dependent], errors="coerce").astype(float)

    data = pd.concat([features, target.rename(dependent)], axis=1).dropna()
    if data.empty:
        print(f"[{label}] all rows dropped after NaN filtering; skipping.")
        return

    X = sm.add_constant(data[available_features], has_constant="add")
    y = data[dependent]

    if len(y) < 2:
        print(f"[{label}] not enough observations; skipping regression.")
        return

    model = sm.OLS(y, X).fit()

    summary_path = dest / "ols_summary.txt"
    summary_path.write_text(model.summary().as_text(), encoding="utf-8")

    sg = Stargazer([model])
    sg.title(f"OLS Regression Results ({dependent})")
    try:
        sg.dependent_variable_name(dependent)
    except Exception:
        pass
    html_path = dest / "ols_stargazer.html"
    html_path.write_text(sg.render_html(), encoding="utf-8")
    tex_path = dest / "ols_stargazer.tex"
    tex_path.write_text(sg.render_latex(), encoding="utf-8")

    print(f"[{label}] wrote {summary_path}")


def process_model(
    df_full: pd.DataFrame,
    model_name: str,
    years: Sequence[str] | None,
    targets: Sequence[str],
    feature_columns: Sequence[str],
    outdir: Path,
    skip_combined: bool,
) -> None:
    df_model = select_rows(df_full, [model_name], years)
    if df_model.empty:
        print(f"[{model_name}] no rows found; skipping model.")
        return

    per_year = years or sorted(df_model["year"].unique())
    for year in per_year:
        df_year = df_model[df_model["year"] == year]
        for target in targets:
            label = f"{model_name}_{year}_{target}"
            dest = outdir / target / model_name / year
            run_regression(df_year, target, feature_columns, dest, label)

    if not skip_combined:
        for target in targets:
            label = f"{model_name}_combined_{target}"
            dest = outdir / target / model_name / "combined"
            run_regression(df_model, target, feature_columns, dest, label)


def process_overall(
    df_full: pd.DataFrame,
    models: Sequence[str] | None,
    years: Sequence[str] | None,
    targets: Sequence[str],
    feature_columns: Sequence[str],
    outdir: Path,
) -> None:
    df_subset = select_rows(df_full, models, years)
    if df_subset.empty:
        print("[all_models] no observations; skipping overall regressions.")
        return

    for target in targets:
        label = f"all_models_{target}"
        dest = outdir / target / "all_models"
        run_regression(df_subset, target, feature_columns, dest, label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regress TDA features against graph properties (OLS)",
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
        "--targets",
        nargs="*",
        default=None,
        help="Graph feature columns to use as dependent variables",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/tda_vs_graph"),
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
    tda_feature_columns = _prefixed_columns(df_full, "tda_")
    if args.targets:
        resolved_targets: list[str] = []
        for target in args.targets:
            if target in df_full.columns:
                resolved_targets.append(target)
            else:
                candidate = f"graph_{target}" if not target.startswith("graph_") else target
                resolved_targets.append(candidate)
        targets = resolved_targets
    else:
        targets = _prefixed_columns(df_full, "graph_")

    requested_models = args.models or sorted(df_full["model"].unique())
    requested_years = [str(year) for year in args.years] if args.years else None

    for model_name in requested_models:
        process_model(
            df_full,
            model_name,
            requested_years,
            targets,
            tda_feature_columns,
            args.outdir,
            args.skip_combined,
        )

    if not args.skip_overall:
        process_overall(
            df_full,
            requested_models,
            requested_years,
            targets,
            tda_feature_columns,
            args.outdir,
        )


if __name__ == "__main__":
    main()
