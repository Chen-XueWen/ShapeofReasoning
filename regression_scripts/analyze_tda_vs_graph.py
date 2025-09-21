#!/usr/bin/env python3
from __future__ import annotations

"""OLS regressions of TDA features against individual graph properties.

For each requested year (and the combined dataset if enabled), this script:
1. Loads the TDA and graph feature JSONL traces and filters rows by year prefix.
2. Merges them on `id` to form a single feature matrix.
3. Runs a separate OLS regression for each graph property listed in
   GRAPH_TARGET_COLUMNS, using the ordered TDA feature set as predictors.
4. Writes statsmodels summaries, Stargazer tables, and coefficient CSVs under
   `<outdir>/<target>/<year>/`.
"""

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

# Ordered TDA features reused from analyze_tda_vs_alignment.py
TDA_FEATURE_COLUMNS = [
    "H0_count",
    "H0_total_life",
    "H0_max_life",
    "H0_mean_life",
    "H0_entropy",
    "H0_skewness",
    "H0_max_birth",
    "H0_max_death",
    "H1_count",
    "H1_total_life",
    "H1_max_life",
    "H1_mean_life",
    "H1_entropy",
    "H1_skewness",
    "H1_max_birth",
    "H1_max_death",
    "H0_betti_peak",
    "H0_betti_location",
    "H0_betti_width",
    "H0_betti_centroid",
    "H0_betti_spread",
    "H0_betti_trend",
    "H1_betti_peak",
    "H1_betti_location",
    "H1_betti_width",
    "H1_betti_centroid",
    "H1_betti_spread",
    "H1_betti_trend",
    "H0_landscape_mean",
    "H0_landscape_max",
    "H0_landscape_area",
    "H1_landscape_mean",
    "H1_landscape_max",
    "H1_landscape_area",
]

# Graph properties to treat as dependent variables by default
GRAPH_TARGET_COLUMNS = [
    "avg_path_length",
    "diameter",
    "loop_count",
    "small_world_index",
    "avg_clustering",
]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No JSONL files found under {path}")
        frames = [pd.read_json(f, lines=True) for f in files]
        return pd.concat(frames, ignore_index=True)
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' not found")
    return pd.read_json(path, lines=True)


def filter_by_year(df: pd.DataFrame, year: str | None) -> pd.DataFrame:
    if year is None:
        return df.copy()
    prefix = f"{year}-"
    return df[df["id"].astype(str).str.startswith(prefix)].copy()


def merge_tda_graph(df_tda: pd.DataFrame, df_graph: pd.DataFrame) -> pd.DataFrame:
    df_tda = df_tda.copy()
    df_graph = df_graph.copy()
    df_tda = df_tda.drop("score", axis=1, errors="ignore")
    df_graph = df_graph.drop("score", axis=1, errors="ignore")

    df_tda["id"] = df_tda["id"].astype(str)
    df_graph["id"] = df_graph["id"].astype(str)

    return df_tda.merge(df_graph, on="id", how="inner")


def run_regression(df: pd.DataFrame, dependent: str, dest: Path, label: str) -> None:
    if dependent not in df.columns:
        print(f"Column '{dependent}' missing for {label}; skipping regression.")
        return

    available_features = [col for col in TDA_FEATURE_COLUMNS if col in df.columns]
    if not available_features:
        print(f"No TDA feature columns present for {label}; skipping regression.")
        return

    ensure_dir(dest)

    features = (
        df[available_features]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )
    target = pd.to_numeric(df[dependent], errors="coerce").astype(float)

    data = pd.concat([features, target.rename(dependent)], axis=1).dropna()
    if data.empty:
        print(f"All rows dropped due to NaNs for {label} ({dependent}); skipping.")
        return

    X = sm.add_constant(data[available_features].astype(float), has_constant="add")
    y = data[dependent].astype(float)

    if len(y) < 2:
        print(f"Not enough observations for regression ({label}, {dependent}).")
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

    print(f"[{label}] {dependent}: wrote {summary_path}")


def process_year(
    df_tda: pd.DataFrame,
    df_graph: pd.DataFrame,
    year: str | None,
    targets: list[str],
    base_outdir: Path,
    label: str,
) -> None:
    df_tda_year = filter_by_year(df_tda, year)
    df_graph_year = filter_by_year(df_graph, year)
    df_year = merge_tda_graph(df_tda_year, df_graph_year)

    if df_year.empty:
        print(f"No merged rows for {label}; skipping year.")
        return

    for target in targets:
        dest = base_outdir / target / label
        run_regression(df_year, target, dest, label)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regress TDA features against graph properties (OLS)"
    )
    ap.add_argument(
        "--tda",
        default="data/aime_tda/trace/gpt-oss_20b.jsonl",
        help="TDA feature JSONL (trace split)",
    )
    ap.add_argument(
        "--graph",
        default="data/aime_graph/trace/gpt-oss_20b.jsonl",
        help="Graph feature JSONL (trace split)",
    )
    ap.add_argument(
        "--outdir",
        default="analysis/gpt-oss_20b/tda_vs_graph",
        help="Directory to write analysis artifacts",
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        default=GRAPH_TARGET_COLUMNS,
        help="Graph feature columns to use as dependent variables",
    )
    ap.add_argument(
        "--years",
        nargs="+",
        default=["2020", "2021", "2022", "2023", "2024", "2025"],
        help="List of AIME contest years to analyze individually",
    )
    ap.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip combined regression across all specified years",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df_tda_full = read_jsonl(Path(args.tda))
    df_graph_full = read_jsonl(Path(args.graph))

    years = [str(y) for y in args.years]
    for year in years:
        process_year(
            df_tda_full,
            df_graph_full,
            year,
            args.targets,
            outdir,
            label=year,
        )

    if not args.skip_combined:
        process_year(
            df_tda_full,
            df_graph_full,
            None,
            args.targets,
            outdir,
            label="combined",
        )


if __name__ == "__main__":
    main()
