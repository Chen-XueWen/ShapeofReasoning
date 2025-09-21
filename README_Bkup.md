TDA on LLM Reasoning Traces
===========================

Pipeline for comparing topological signatures of LLM reasoning traces against AIME gold solutions. Scripts cover: dataset collection, trace generation (via Ollama), step segmentation + embeddings, TDA feature extraction, and optional alignment to gold steps.

What you get
-------------
- End-to-end scripts in `scripts/` for each stage
- Heuristic step segmentation + sentence-transformer embeddings
- Vietoris–Rips persistent homology (ripser) + derived features (Betti curves, summary stats)
- Alignment of model steps to gold steps

Prerequisites
-------------
- Python 3.10+
- For generation: Ollama running locally and a pulled model (e.g., `gpt-oss:20b`)
- Python deps (install as needed):
  - `pip install -U sentence-transformers ripser datasets huggingface_hub polars requests tqdm`

Quickstart
----------
1) Collect AIME 2024 data from HF Hub
   - Example (defaults target Maxwell-Jia/AIME_2024):
     `python scripts/collect_aime.py --dataset Maxwell-Jia/AIME_2024 --out data/raw/aime2024.jsonl`

2) Generate one reasoning trace per problem (Ollama)
   - Ensure `ollama serve` is running and the model is available:
     `ollama pull gpt-oss:20b`
   - Run:
     `python scripts/generate_traces.py --aime data/raw/aime2024.jsonl --model gpt-oss:20b`
   - Output: `data/raw/traces_aime2024.jsonl`

3) Segment steps and compute embeddings
   - Uses sentence-transformers (default: `sentence-transformers/all-MiniLM-L6-v2`)
   - Run:
     `python scripts/embed_steps.py --traces data/raw/traces_aime2024.jsonl --outdir data/processed/embeddings --model-name sentence-transformers/all-MiniLM-L6-v2`
   - Output: one `.npz` per problem ID in `data/processed/embeddings/`

4) Extract TDA features
   - Uses ripser for VR diagrams (default metric: cosine, maxdim: 1)
   - Run:
     `python regression_scripts/create_diags.py --emb-dir data/processed/embeddings --diag-dir data/processed/diagrams --features-out data/processed/tda_features.jsonl`
   - Output: diagrams saved under `data/processed/diagrams/` and `data/processed/tda_features.jsonl`

5) Align model steps to gold steps
   - If dataset includes structured gold steps they are used; otherwise gold solutions are segmented heuristically
   - Run:
     `python scripts/align_to_gold.py --aime data/raw/aime2024.jsonl --traces data/raw/traces_aime2024.jsonl --out data/processed/alignments.jsonl`
   - Output: `data/processed/alignments.jsonl`

Repository layout
-----------------
- `scripts/` — CLI entry points for each stage
- `tda_reasoning/embedding/` — step segmentation + embedding wrapper
- `tda_reasoning/tda/` — filtrations, persistence, and feature assembly
- `tda_reasoning/eval/` — simple step alignment utilities
- `configs/default.yaml` — reference defaults (not auto-loaded). If you prefer, keep a copy under `data/configs/default.yaml`.
- `data/raw/` — inputs (HF-normalized AIME, generated traces)
- `data/processed/` — outputs (embeddings, TDA features, alignments)

Config reference (defaults)
---------------------------
- Generation (Ollama): model `gpt-oss:20b`, `temperature=0.0`, `seed=7`, `num_predict=512`

**Batch Experiments**
- Purpose: run the full pipeline across multiple models and write per-model analyses.
- Command: `bash run_experiments.sh`
- Models: `qwen3:8b`, `qwen3:30b`, `deepseek-r1:8b`, `deepseek-r1:8b`, `gpt-oss:20b`, `gpt-oss:120b`, `qwen3:235b`
- Outputs per model (slug is the model name with `:/` replaced by `_`):
  - `data/raw/traces_<slug>.jsonl`
  - `data/processed/<slug>/embeddings/`
  - `data/processed/<slug>/tda_features.jsonl`
  - `data/processed/<slug>/alignments.jsonl`
  - `data/processed/analysis/<slug>/` (OLS summary, coefficients, Stargazer)
- Prerequisites: ensure models are pulled in Ollama, for example:
  - `ollama pull qwen3:8b`
  - `ollama pull qwen3:30b`
  - `ollama pull deepseek-r1:8b`
  - `ollama pull gpt-oss:20b`
  - `ollama pull gpt-oss:120b`
  - `ollama pull qwen3:235b`
  - Note: `deepseek-r1:8b` appears twice in the list; running twice will overwrite outputs.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (configurable via `--model-name`)
- TDA: Vietoris–Rips on cosine distances, `maxdim=1` with summary stats and Betti curves

Config notes
------------
- Example config at `configs/default.yaml` reflects the simplified TDA setup (no persistence images/landscapes). You can mirror it to `data/configs/default.yaml` if you keep configs alongside data.

Tips & troubleshooting
----------------------
- Ollama errors: ensure `ollama serve` is running and the model is pulled.
- Missing deps: install `sentence-transformers` and `ripser`.
- HF dataset access: the collector tries `datasets`; if unavailable it falls back to `huggingface_hub + polars`.
- Reproducibility: generation uses `temperature=0` and a fixed `--seed`.

Notes
-----
- Filtration currently supports VR up to H1; more can be added.
- All scripts write under `data/` for reproducible runs and easy diffing.


Betti curves features:
--------------------------------------

betti_peak: max Betti count.
betti_location: normalized location of the peak in [0,1].
betti_width: normalized FWHM (full width at half max) in [0,1].
betti_centroid: normalized first moment in [0,1].
betti_spread: normalized std around centroid in [0,1].
betti_trend: Pearson correlation between t and Betti values in [-1,1].


Multicollinearity Analysis (VIF + PCA)
--------------------------------------
- Purpose: Many topological features are correlated. We report multicollinearity and offer an orthogonalized regression via PCA.

- Outputs (per run under the same analysis folder as OLS):
  - `vif.csv`: Variance Inflation Factor per predictor (controls + features). Higher VIF indicates stronger multicollinearity; common flags are VIF > 5–10.
  - `pca_explained_variance.csv`: Per‑component explained variance ratio and cumulative ratio for the top K principal components (default K=10). The first two PCs typically summarize a large share of variation; the cumulative ratio gives the “XX% of the variance” figure.
  - `pca_ols_summary.txt`: OLS summary when regressing `score` on the first K PCs (default K=2) plus controls (coverage, n_pairs if available).
  - `pca_ols_coefficients.csv`: Coefficients for the PC regression (PCs only).
  - `pca_ols_stargazer.html/.tex`: Stargazer‑formatted tables for the PCA regression.
  - `pca_loadings.csv`: Feature-by-PC loadings (weights of each standardized feature in each PC).

- CLI options (scripts/analyze_tda_vs_alignment.py):
  - `--pca-reg-k`: number of principal components used in the regression (default 2).
  - `--pca-report-k`: number of top PCs to include in the explained‑variance report (default 10).

- Interpretation tips:
  - VIF: values above ~5 suggest notable multicollinearity; above ~10 are strong. Consider PCA‑OLS or regularization when VIF is high.
  - PCA explained variance: the cumulative ratio for PC1..PCk is the “XX% of the variance” covered by the retained components.
  - PCA‑OLS: uses orthogonal components; coefficients are not directly interpretable as original features, but inference is more stable under collinearity.

PCA loadings
---------------------------------------------
- Definition: loadings are the coefficients of each principal component when features are z‑scored. If `z_j` is feature j standardized (mean 0, std 1), then
  - `PC_i = Σ_j loading[j, i] * z_j` (column `PCi` gives the weights for component i).
- Magnitude: larger absolute values indicate stronger contribution of that feature to the component.
- Sign: features with the same sign on a PC tend to increase together along that direction; opposite signs indicate inverse movement.
- Alignment: columns `PC1..PCk` line up with rows in `pca_explained_variance.csv` (same ordering), so you can relate a PC’s variance share to its dominant features via the loadings.

## TO DO LIST:
1. Change max token
2. Check OLS and PCA OLS 
3. Check the betti location
4. Understand the DP
