#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze association between topological features and alignment score using OLS.

Configuration:
- Dependent variable: score
- Predictors: all numeric TDA features (excluding id/score/coverage/n_pairs/indices),
  including scalar Betti descriptors (betti_*)
- Controls: coverage, n_pairs (if available)

Outputs (default under data/processed/analysis/):
- ols_summary.txt: OLS regression summary
- ols_coefficients.csv: per-feature OLS coefficients with p-values
- ols_stargazer.html: Stargazer-formatted regression table (HTML)
- ols_stargazer.tex: Stargazer-formatted regression table (LaTeX)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_data(align_path: Path, tda_path: Path) -> pd.DataFrame:
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
    cols = ["id", "score", "coverage", "n_pairs", "indices"]
    df_align = df_align[[c for c in cols if c in df_align.columns]]

    df = df_feats.merge(df_align, on="id", how="inner")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze TDA feature influence on alignment score (OLS)")
    ap.add_argument("--align", default="data/processed/alignments.jsonl")
    ap.add_argument("--features", default="data/processed/tda_features.jsonl")
    ap.add_argument("--outdir", default="data/processed/analysis")
    ap.add_argument("--pca-reg-k", type=int, default=5, help="Number of principal components for PCA-OLS")
    ap.add_argument("--pca-report-k", type=int, default=10, help="Top-K PCs to report explained variance")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = load_data(Path(args.align), Path(args.features))

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
        if c not in {"id", "score", "coverage", "n_pairs", "indices"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if betti_cols:
        included_betti = [c for c in betti_cols if c in feat_cols]
        if included_betti:
            print(f"Including Betti descriptors: {included_betti}")

    feature_terms = feat_cols
    controls = [c for c in ["coverage", "n_pairs"] if c in df.columns]

    # OLS on score
    X = df[controls + feature_terms].astype(float) if feature_terms else df[controls].astype(float)
    X = sm.add_constant(X)
    y_raw = df["score"].astype(float)
    y = pd.Series(y_raw.to_numpy(), index=y_raw.index, name="score")
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
    sg.title("OLS Regression Results (score)")
    try:
        sg.dependent_variable_name("score")
    except Exception:
        pass
    html_path = outdir / "ols_stargazer.html"
    html_path.write_text(sg.render_html(), encoding="utf-8")
    print(f"Wrote: {html_path}")
    tex_path = outdir / "ols_stargazer.tex"
    tex_path.write_text(sg.render_latex(), encoding="utf-8")
    print(f"Wrote: {tex_path}")

    # VIF report (no NaNs assumed by user; compute on predictors excluding constant)
    X_vif_df = df[controls + feature_terms].astype(float)
    vif_rows = []
    for i, col in enumerate(X_vif_df.columns):
        vif_val = float(variance_inflation_factor(X_vif_df.values, i))
        vif_rows.append({"feature": col, "VIF": vif_val})
    vif_out = outdir / "vif.csv"
    pd.DataFrame(vif_rows).to_csv(vif_out, index=False)
    print(f"Wrote: {vif_out}")

    # PCA on feature_terms, then OLS on PCs + controls
    F = df[feature_terms].astype(float)
    F_mat = F.to_numpy()
    # standardize
    mu = F_mat.mean(axis=0)
    sigma = F_mat.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (F_mat - mu) / sigma
    # SVD for PCA
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2
    total_var = float(var.sum()) if var.size else 0.0
    if total_var > 0:
        expl = var / total_var
        cum_expl = np.cumsum(expl)
    else:
        expl = np.zeros_like(var)
        cum_expl = np.zeros_like(var)
    k_rep = int(min(args.pca_report_k, len(expl)))
    rep_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(k_rep)],
        "explained_variance_ratio": expl[:k_rep],
        "cumulative_variance_ratio": cum_expl[:k_rep],
    })
    ev_out = outdir / "pca_explained_variance.csv"
    rep_df.to_csv(ev_out, index=False)
    print(f"Wrote: {ev_out}")

    # Scores for all rows
    scores = Z @ Vt.T  # (n_samples, n_components)
    # Loadings table: feature-by-PC coefficients (unit-norm eigenvectors on standardized features)
    k_load = int(min(len(F.columns), max(args.pca_report_k, args.pca_reg_k)))
    load_cols = [f"PC{i+1}" for i in range(k_load)]
    load_df = pd.DataFrame(Vt.T[:, :k_load], index=F.columns, columns=load_cols)
    load_out = outdir / "pca_loadings.csv"
    load_df.to_csv(load_out)
    print(f"Wrote: {load_out}")
    k_reg = int(min(args.pca_reg_k, scores.shape[1]))
    pc_cols = [f"PC{i+1}" for i in range(k_reg)]
    pcs_df = pd.DataFrame(scores[:, :k_reg], columns=pc_cols, index=F.index)

    # PCA-OLS with controls
    X_pca = pd.concat([df[controls].astype(float), pcs_df], axis=1) if controls else pcs_df
    X_pca = sm.add_constant(X_pca)
    model_pca = sm.OLS(y, X_pca, missing="drop").fit()
    with (outdir / "pca_ols_summary.txt").open("w", encoding="utf-8") as f:
        f.write(model_pca.summary().as_text())

    # Export PC coefficients only
    params_p = model_pca.params
    bse_p = model_pca.bse
    tvals_p = model_pca.tvalues
    pvals_p = model_pca.pvalues
    ci_p = model_pca.conf_int()
    rows_p = []
    for feat in pc_cols:
        if feat in params_p.index:
            rows_p.append({
                "feature": feat,
                "coef": float(params_p[feat]),
                "std_err": float(bse_p[feat]),
                "t": float(tvals_p[feat]),
                "pval": float(pvals_p[feat]),
                "ci_low": float(ci_p.loc[feat, 0]),
                "ci_high": float(ci_p.loc[feat, 1]),
            })
    pca_coef_df = pd.DataFrame(rows_p).sort_values(by="pval", ascending=True)
    pca_coef_out = outdir / "pca_ols_coefficients.csv"
    pca_coef_df.to_csv(pca_coef_out, index=False)
    print(f"Wrote: {pca_coef_out}")

    # Stargazer for PCA-OLS
    sg2 = Stargazer([model_pca])
    sg2.title("OLS Regression Results (PCs + controls)")
    try:
        sg2.dependent_variable_name("score")
    except Exception:
        pass
    html2 = outdir / "pca_ols_stargazer.html"
    html2.write_text(sg2.render_html(), encoding="utf-8")
    print(f"Wrote: {html2}")
    tex2 = outdir / "pca_ols_stargazer.tex"
    tex2.write_text(sg2.render_latex(), encoding="utf-8")
    print(f"Wrote: {tex2}")


if __name__ == "__main__":
    main()
