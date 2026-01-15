# TDA on LLM Reasoning Traces

Pipeline for comparing topological signatures of LLM reasoning traces against AIME gold solutions, as described in the paper "2510.20665v1.pdf".

This repository contains scripts for:

1. **Trace Generation**: Generating reasoning traces using Ollama.
2. **Alignment & Embedding**: Segmenting steps, computing embeddings (SentenceTransformer), and aligning model steps to gold steps.
3. **Feature Extraction**: Extracting Topological Data Analysis (TDA) features (Betti curves) and Graph features.
4. **Analysis**: Running regression analysis to compare topological signatures with alignment scores.

## Prerequisites

- **Python 3.10+**
- **Ollama**: Must be running locally for trace generation.
  - Install Ollama: [https://ollama.com/](https://ollama.com/)
  - Pull required models (e.g., `ollama pull gpt-oss:20b`)

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

*Key libraries*: `numpy`, `sentence-transformers`, `ripser`, `tqdm`, `requests`, `pandas`, `statsmodels`, `stargazer`.

## Repository Layout

- `generate_traces.py`: Script to generate reasoning traces from AIME problems.
- `feature_scripts/`: Scripts for feature extraction and alignment.
  - `align_to_gold.py`: Aligns model traces to gold solutions and generates embeddings.
  - `tda_extract.py`: Extracts TDA features (Betti curves) from embeddings.
  - `graph_extract.py`: Extracts graph-based features.
- `regression_scripts/`: Scripts for statistical analysis and regression formulations.
- `scripts/`: (Optional) Helper scripts if present.
- `data/`: Directory for storing raw data, traces, processed embeddings, and analysis results.

## Quickstart Pipeline

### 1. Trace Generation

Generate reasoning traces for a specific model and year (defaults to all years in `data/aime_json` if using the shell script).

**Using the shell script (recommended):**

```bash
./run_generate_traces.sh
```

**Manual usage:**

```bash
python generate_traces.py --aime data/aime_json/aime2024.json --out data/aime_traces/gpt-oss_20b/traces_aime2024.jsonl --model gpt-oss:20b
```

### 2. Alignment & Embedding

Align the generated traces to the gold solutions. This step also computes and saves the embeddings for both the trace and the gold solution.

**Using the shell script:**

```bash
./run_align_all.sh
```

**Manual usage:**

```bash
python feature_scripts/align_to_gold.py \
    --traces data/aime_traces/gpt-oss_20b/traces_aime2024.jsonl \
    --out data/aime_align_dp/gpt-oss_20b/align_aime2024.jsonl \
    --embed-outdir data/aime_embed/gpt-oss_20b/
```

### 3. Feature Extraction

Extract topological and graph features from the embeddings.

**TDA Extraction:**

```bash
./run_tda_extract_all.sh
```

**Graph Extraction:**

```bash
./run_graph_extract_all.sh
```

### 4. Analysis & Regression

Run the full suite of regression analyses to compare TDA/Graph features against alignment scores.

```bash
./run_regressions_all.sh
```

This will output analysis results (coefficients, summaries, Stargazer tables) to the `analysis/` directory.

## Configuration & Notes

### Models

The scripts are configured to run on a set of models defined in the `.sh` files (e.g., `deepseek-r1:32b`, `gpt-oss:20b`). Edit the `MODELS` array in the shell scripts to add or remove models.

### TDA Features (Betti Curves)

The pipeline computes several summary statistics from the Betti curves:

- `betti_peak`: Max Betti count.
- `betti_location`: Normalized location of the peak [0,1].
- `betti_width`: Normalized full width at half max (FWHM).
- `betti_centroid`: Normalized first moment.
- `betti_spread`: Normalized std dev around centroid.
- `betti_trend`: Pearson correlation between parameter `t` and Betti values.

### Multicollinearity Analysis

The analysis scripts (`analyze_*.py`) handle multicollinearity in topological features using PCA.

- **VIF**: Reported to identify high collinearity.
- **PCA-OLS**: Performs regression on Principal Components of the features rather than raw features.
