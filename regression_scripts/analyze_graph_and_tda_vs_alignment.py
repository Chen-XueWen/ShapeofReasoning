#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze associations between combined graph + TDA features and alignment scores using OLS.

Configuration:
- Dependent variable: score
- Predictors: ordered set of TDA + graph features (see COMBINED_FEATURE_COLUMNS)

Outputs (default under analysis/):
- ols_summary.txt: OLS regression summary
- ols_coefficients.csv: per-feature OLS coefficients with p-values
- ols_stargazer.html / .tex: Stargazer-formatted regression tables
"""

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

# Ordered feature lists
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

GRAPH_FEATURE_COLUMNS = [
    "has_loop",
    "loop_count",
    "diameter",
    "avg_clustering",
    "avg_path_length",
    "small_world_index",
]

COMBINED_FEATURE_COLUMNS = TDA_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS


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


def load_align_df(align_path: Path, year: str | None) -> pd.DataFrame:
    """Load alignment data for a specific year or across all years."""

    if align_path.is_dir():
        if year is None:
            files = sorted(align_path.glob("*.jsonl"))
            if not files:
                raise FileNotFoundError(f"No alignment JSONL files found under {align_path}")
        else:
            pattern = f"*{year}*.jsonl"
            files = sorted(align_path.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No alignment file matching '{pattern}' found under {align_path}"
                )
        frames = [pd.read_json(f, lines=True) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = read_jsonl(align_path)
        if year is not None:
            df = df[df["id"].astype(str).str.startswith(f"{year}")].copy()

    return df


def filter_features_by_year(df_feats: pd.DataFrame, year: str | None) -> pd.DataFrame:
    if year is None:
        return df_feats.copy()
    prefix = f"{year}-"
    return df_feats[df_feats["id"].astype(str).str.startswith(prefix)].copy()


def merge_feature_sources(df_tda: pd.DataFrame, df_graph: pd.DataFrame) -> pd.DataFrame:
    df_tda = df_tda.copy()
    df_graph = df_graph.copy()
    df_tda = df_tda.drop("score", axis=1, errors="ignore")
    df_graph = df_graph.drop("score", axis=1, errors="ignore")
    df_tda["id"] = df_tda["id"].astype(str)
    df_graph["id"] = df_graph["id"].astype(str)
    return df_tda.merge(df_graph, on="id", how="inner")


def merge_align_features(df_align: pd.DataFrame, df_feats: pd.DataFrame) -> pd.DataFrame:
    df_align = df_align.copy()
    df_feats = df_feats.copy()

    df_align["id"] = df_align["id"].astype(str)
    df_feats["id"] = df_feats["id"].astype(str)

    if "indices" in df_align.columns:
        df_align["n_pairs"] = df_align["indices"].apply(
            lambda x: len(x) if isinstance(x, list) else pd.NA
        )
    else:
        df_align["n_pairs"] = pd.NA

    cols = ["id", "score", "coverage", "n_pairs", "indices"]
    df_align = df_align[[c for c in cols if c in df_align.columns]]

    return df_feats.merge(df_align, on="id", how="inner")


def run_analysis(df: pd.DataFrame, dest: Path, label: str) -> None:
    if df.empty:
        print(f"No merged rows available for {label}; skipping analysis.")
        return

    ensure_dir(dest)

    available_features = [col for col in COMBINED_FEATURE_COLUMNS if col in df.columns]
    if not available_features:
        print(f"No configured feature columns present for {label}; skipping analysis.")
        return

    features = (
        df[available_features]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )
    if "score" not in df.columns:
        print(f"Column 'score' missing for {label}; skipping analysis.")
        return
    target = pd.to_numeric(df["score"], errors="coerce").astype(float)

    data = pd.concat([features, target.rename("score")], axis=1).dropna()

    if data.empty:
        print(f"All rows dropped due to NaNs for {label}; skipping analysis.")
        return

    X = sm.add_constant(data[available_features].astype(float), has_constant="add")
    y = data["score"].astype(float)

    if len(y) < 2:
        print(f"Not enough observations for regression ({label}); skipping analysis.")
        return

    model = sm.OLS(y, X).fit()

    summary_path = dest / "ols_summary.txt"
    summary_path.write_text(model.summary().as_text(), encoding="utf-8")

    sg = Stargazer([model])
    sg.title("OLS Regression Results (score)")
    try:
        sg.dependent_variable_name("score")
    except Exception:
        pass
    html_path = dest / "ols_stargazer.html"
    html_path.write_text(sg.render_html(), encoding="utf-8")
    tex_path = dest / "ols_stargazer.tex"
    tex_path.write_text(sg.render_latex(), encoding="utf-8")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {html_path}")
    print(f"Wrote: {tex_path}")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze combined graph + TDA feature influence on alignment score (OLS)"
    )
    ap.add_argument(
        "--align",
        default="data/aime_align_dp/gpt-oss_20b",
        help="Alignment JSONL file or directory",
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
        default="analysis/gpt-oss_20b/graph_and_tda_vs_alignment",
        help="Directory to write analysis artifacts",
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

    align_path = Path(args.align)
    tda_path = Path(args.tda)
    graph_path = Path(args.graph)

    df_tda_full = read_jsonl(tda_path)
    df_graph_full = read_jsonl(graph_path)

    years = [str(y) for y in args.years]

    for year in years:
        df_tda_year = filter_features_by_year(df_tda_full, year)
        df_graph_year = filter_features_by_year(df_graph_full, year)
        df_feats_year = merge_feature_sources(df_tda_year, df_graph_year)

        df_align_year = load_align_df(align_path, year)
        df_year = merge_align_features(df_align_year, df_feats_year)
        year_outdir = outdir / year
        run_analysis(df_year, year_outdir, year)

    if not args.skip_combined:
        df_feats_all = merge_feature_sources(df_tda_full, df_graph_full)
        df_align_all = load_align_df(align_path, None)
        df_all = merge_align_features(df_align_all, df_feats_all)
        run_analysis(df_all, outdir / "combined", "combined")


if __name__ == "__main__":
    main()
