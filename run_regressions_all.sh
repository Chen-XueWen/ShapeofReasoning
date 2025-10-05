#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

REGRESSION_DIR="regression_scripts"
SCRIPTS=(
  "analyze_tda_vs_alignment.py"
  "analyze_graph_vs_alignment.py"
  "analyze_graph_and_tda_vs_alignment.py"
  "analyze_tda_vs_graph.py"
)

for script in "${SCRIPTS[@]}"; do
  if [[ ! -f "${REGRESSION_DIR}/${script}" ]]; then
    echo "Required script '${REGRESSION_DIR}/${script}' not found" >&2
    exit 1
  fi
done

MODELS=(
  #"gpt-oss:120b"
  #"deepseek-r1:32b"
  #"qwen3:32b"
  #"gpt-oss:20b"
  "deepseek-r1:8b"
  "qwen3:8b"
)

sanitize() {
  local name=$1
  printf '%s' "${name//[^A-Za-z0-9._-]/_}"
}

for model in "${MODELS[@]}"; do
  safe_model=$(sanitize "${model}")

  align_root="data/aime_align_dp/${safe_model}"
  tda_path="data/aime_tda/trace/${safe_model}.jsonl"
  graph_path="data/aime_graph/trace/${safe_model}.jsonl"

  if [[ ! -d "${align_root}" ]]; then
    echo "Alignment directory '${align_root}' not found" >&2
    exit 1
  fi
  if [[ ! -f "${tda_path}" ]]; then
    echo "TDA feature file '${tda_path}' not found" >&2
    exit 1
  fi
  if [[ ! -f "${graph_path}" ]]; then
    echo "Graph feature file '${graph_path}' not found" >&2
    exit 1
  fi

  echo "Running regressions for model '${model}'"

  ${PYTHON_BIN} "${REGRESSION_DIR}/analyze_tda_vs_alignment.py" \
    --align "${align_root}" \
    --features "${tda_path}" \
    --outdir "analysis/${safe_model}/tda_vs_alignment"

  ${PYTHON_BIN} "${REGRESSION_DIR}/analyze_graph_vs_alignment.py" \
    --align "${align_root}" \
    --features "${graph_path}" \
    --outdir "analysis/${safe_model}/graph_vs_alignment"

  ${PYTHON_BIN} "${REGRESSION_DIR}/analyze_graph_and_tda_vs_alignment.py" \
    --align "${align_root}" \
    --tda "${tda_path}" \
    --graph "${graph_path}" \
    --outdir "analysis/${safe_model}/graph_and_tda_vs_alignment"

  ${PYTHON_BIN} "${REGRESSION_DIR}/analyze_tda_vs_graph.py" \
    --tda "${tda_path}" \
    --graph "${graph_path}" \
    --outdir "analysis/${safe_model}/tda_vs_graph"

done

echo "All regressions completed."
