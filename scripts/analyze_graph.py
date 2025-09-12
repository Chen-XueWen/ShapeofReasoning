#!/usr/bin/env python3
from __future__ import annotations


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_data(align_path: Path, tda_path: Path, dependent_variable="small_world_index") -> pd.DataFrame:
    df_align = pd.read_json(align_path, lines=True)
    df_feats = pd.read_json(tda_path, lines=True)
    # Ensure id is string for both
    df_align["id"] = df_align["id"].astype(str)
    df_feats["id"] = df_feats["id"].astype(str)

    # Extract controls
    if "indices" in df_align.columns:
        df_align["n_pairs"] = df_align["indices"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)
    else:
        df_align["n_pairs"] = np.nan
    # Keep only relevant columns (tolerant if some are missing)
    cols = ["id", dependent_variable, "coverage", "n_pairs", "indices"]
    df_align = df_align[[c for c in cols if c in df_align.columns]]

    df = df_feats.merge(df_align, on="id", how="inner")
    return df

def check_multicollinearity(df, outdir=Path("data/processed/analysis")):
    """Check for multicollinearity using correlation matrix"""

    # Get numeric features only
    feature_cols = df.select_dtypes(include=[np.number]).columns 

    with (outdir / "multicollinearity_analysis.txt").open("w", encoding="utf-8") as f:
        
        if len(feature_cols) < 2:
            f.write("Not enough numeric features for multicollinearity analysis\n")
            return
        
        # Correlation matrix analysis
        corr_matrix = df[feature_cols].corr()

        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            for var1, var2, corr_val in high_corr_pairs:
                f.write(f"  {var1} <-> {var2}: {corr_val:.3f}\n")
            # print("\nConsider removing one variable from each highly correlated pair.")
        else:
            f.write("✓ No problematic correlations detected (all <0.8)\n")

def main() -> None:
    dependent_variables = ["small_world_index", "avg_path_length", "avg_clustering", "diameter", "loop_count"]
    for variable in dependent_variables:
        dependent_variable = variable  # Change to desired dependent variable

        ap = argparse.ArgumentParser(description="Analyze TDA feature influence on graph_features (OLS)")
        # ap.add_argument("--align", default="data/processed/alignments.jsonl")
        # ap.add_argument("--features", default="data/processed/tda_features.jsonl")
        ap.add_argument("--align", default="data/processed/graph_analysis.jsonl")
        ap.add_argument("--features", default="data/processed/gpt-oss_20b_1/tda_features.jsonl")
        ap.add_argument("--outdir", default=f"data/processed/analysis/{dependent_variable}")

        args = ap.parse_args()

        outdir = Path(args.outdir)
        ensure_dir(outdir)


        df = load_data(Path(args.align), Path(args.features), dependent_variable)

        # Prefer scalar Betti descriptors over raw BC vector; ensure numeric dtype
        if "BC" in df.columns:
            print("Dropping raw 'BC' vector column; using betti_* scalar descriptors instead.")
            df = df.drop(columns=["BC"])  # old curve array not used in OLS
        betti_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("betti_")]
        for c in betti_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Identify feature columns: numeric TDA columns excluding id and control columns
        feat_cols = [
            c for c in df.columns
            if c not in {"id", dependent_variable, "coverage", "n_pairs", "indices", "H0_betti_location", "H0_betti_peak", "H0_betti_trend", "H1_betti_trend", "H0_mean_life", "H0_skewness", "H1_mean_life", "H1_skewness"}
            #including H1_count, H0_entropy
            and pd.api.types.is_numeric_dtype(df[c])
            # and c in {"H0_total_life", "H0_max_life", "H0_entropy", "H1_entropy"}
        ] ###TODO: test this and try it out!!!!!
        if betti_cols:
            included_betti = [c for c in betti_cols if c in feat_cols]
            if included_betti:
                print(f"Including Betti descriptors: {included_betti}")

        feature_terms = feat_cols
        controls = [c for c in ["coverage", "n_pairs"] if c in df.columns]

        # OLS on score
        # X = df[controls + feature_terms].astype(float) if feature_terms else df[controls].astype(float)

        X = df[feature_terms].astype(float) 
        check_multicollinearity(X, outdir)
        X = sm.add_constant(X)
        y_raw = df[dependent_variable].astype(float)
        y = pd.Series(y_raw.to_numpy(), index=y_raw.index, name=dependent_variable)
        model = sm.OLS(y, X, missing="drop").fit()
        with (outdir / "ols_summary.txt").open("w", encoding="utf-8") as f:
            f.write(model.summary().as_text())

        # Export features-only coefficients for significance checks
        params = model.params
        bse = model.bse
        tvals = model.tvalues
        pvals = model.pvalues
        ci = model.conf_int()
        rows = []
        for feat in feature_terms:
            if feat in params.index:
                rows.append({
                    "feature": feat,
                    "coef": float(params[feat]),
                    "std_err": float(bse[feat]),
                    "t": float(tvals[feat]),
                    "pval": float(pvals[feat]),
                    "ci_low": float(ci.loc[feat, 0]),
                    "ci_high": float(ci.loc[feat, 1]),
                })
        coef_df = pd.DataFrame(rows).sort_values(by="pval", ascending=True)
        coef_out = outdir / "ols_coefficients.csv"
        coef_df.to_csv(coef_out, index=False)

        if (outdir / "ols_summary.txt").exists():
            print(f"Wrote: {(outdir / 'ols_summary.txt')}")
        print(f"Wrote: {coef_out}")

        sg = Stargazer([model])
        sg.title("OLS Regression Results" + dependent_variable)
        try:
            sg.dependent_variable_name(dependent_variable)
        except Exception:
            pass
        html_path = outdir / "ols_stargazer.html"
        html_path.write_text(sg.render_html(), encoding="utf-8")
        print(f"Wrote: {html_path}")
        tex_path = outdir / "ols_stargazer.tex"
        tex_path.write_text(sg.render_latex(), encoding="utf-8")
        print(f"Wrote: {tex_path}")

        # VIF report (no NaNs assumed by user; compute on predictors excluding constant)
        # X_vif_df = df[controls + feature_terms].astype(float)
        X_vif_df = df[feature_terms].astype(float)
        vif_rows = []
        for i, col in enumerate(X_vif_df.columns):
            vif_val = float(variance_inflation_factor(X_vif_df.values, i))
            vif_rows.append({"feature": col, "VIF": vif_val})
        vif_out = outdir / "vif.csv"
        pd.DataFrame(vif_rows).to_csv(vif_out, index=False)
        print(f"Wrote: {vif_out}")



if __name__ == "__main__":
    main()
