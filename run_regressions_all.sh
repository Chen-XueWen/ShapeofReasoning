#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

REGRESSION_DIR="regression_scripts"
DATASET_PATH="data/aime_regression_dataset_cleaned.csv"

SCRIPTS=(
  "analyze_tda_vs_alignment.py"
  "analyze_graph_vs_alignment.py"
  "analyze_graph_and_tda_vs_alignment.py"
  "analyze_tda_vs_graph.py"
)

MODELS=(
  #"gpt-oss_120b"
  #"deepseek-r1_32b"
  #"qwen3_32b"
  #"gpt-oss_20b"
  #"deepseek-r1_8b"
  #"deepseek-r1_70b"
  "qwen3_8b"
)

run_regression() {
  local script=$1
  local outdir
  case "${script}" in
    "analyze_tda_vs_alignment.py")
      outdir="analysis/tda_vs_alignment"
      ;;
    "analyze_graph_vs_alignment.py")
      outdir="analysis/graph_vs_alignment"
      ;;
    "analyze_graph_and_tda_vs_alignment.py")
      outdir="analysis/graph_and_tda_vs_alignment"
      ;;
    "analyze_tda_vs_graph.py")
      outdir="analysis/tda_vs_graph"
      ;;
    *)
      echo "No output directory mapping for '${script}'" >&2
      exit 1
      ;;
  esac

  echo "Running ${DATASET_PATH} ${script} -> ${outdir}"
  "${PYTHON_BIN}" "${REGRESSION_DIR}/${script}" \
    --dataset "${DATASET_PATH}" \
    --models "${MODELS[@]}" \
    --outdir "${outdir}"
}

for script in "${SCRIPTS[@]}"; do
  run_regression "${script}"
done

echo "All regressions completed. Outputs written under 'analysis/'."
