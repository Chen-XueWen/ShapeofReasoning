#!/usr/bin/env bash

set -euo pipefail

# Models whose traces you want to align. Extend the list as needed.
MODELS=(
  #"gpt-oss:120b"
  #"gpt-oss:20b"
  #"deepseek-r1_32b"
  #"qwen3_32b"
  "deepseek-r1:8b"
  "qwen3:8b"
)

TRACE_ROOT="./data/aime_traces"
ALIGN_ROOT="./data/aime_align_dp"
ALIGN_SCRIPT="./feature_scripts/align_to_gold.py"

if [[ ! -f "${ALIGN_SCRIPT}" ]]; then
  echo "Alignment script '${ALIGN_SCRIPT}' not found" >&2
  exit 1
fi

sanitize() {
  local name=$1
  printf '%s' "${name//[^A-Za-z0-9._-]/_}"
}

for MODEL in "${MODELS[@]}"; do
  safe_model=$(sanitize "${MODEL}")
  TRACE_DIR="${TRACE_ROOT}/${safe_model}"
  OUT_DIR="${ALIGN_ROOT}/${safe_model}"
  EMBED_DIR="./data/aime_embed/${safe_model}"

  if [[ ! -d "${TRACE_DIR}" ]]; then
    echo "Trace directory '${TRACE_DIR}' not found" >&2
    exit 1
  fi

  mkdir -p "${OUT_DIR}"

  shopt -s nullglob
  trace_files=("${TRACE_DIR}"/traces_*.jsonl)
  shopt -u nullglob

  if [[ ${#trace_files[@]} -eq 0 ]]; then
    echo "No trace files matching '${TRACE_DIR}/traces_*.jsonl'" >&2
    exit 1
  fi

  for trace_file in "${trace_files[@]}"; do
    base_name=$(basename "${trace_file}" ".jsonl")
    suffix=${base_name#traces_}
    out_file="${OUT_DIR}/align_${suffix}.jsonl"
    echo "Aligning model '${MODEL}' file '${trace_file}' -> '${out_file}'"
    python3 "${ALIGN_SCRIPT}" \
      --traces "${trace_file}" \
      --out "${out_file}" \
      --embed-outdir "${EMBED_DIR}" \
      "$@"
  done
done

echo "All alignment jobs completed."
