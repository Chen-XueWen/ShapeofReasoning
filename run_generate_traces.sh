#!/usr/bin/env bash

set -euo pipefail

# Directory containing the per-year AIME JSON inputs
JSON_DIR="./data/aime_json"

# Python script that generates traces
SCRIPT="./scripts/generate_traces.py"

if [[ ! -d "${JSON_DIR}" ]]; then
  echo "Input directory '${JSON_DIR}' not found" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Generator script '${SCRIPT}' not found" >&2
  exit 1
fi

models=(
  #"deepseek-r1:32b"
  #"gpt-oss:20b"
  #"gpt-oss:120b"
  #"deepseek-r1:8b"
  #"deepseek-r1:70b"
  #"qwen3:8b"
  "qwen3:32b"
  #"qwen3:235b"
)

shopt -s nullglob
inputs=("${JSON_DIR}"/aime*.json)
shopt -u nullglob

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "No input files matching '${JSON_DIR}/aime*.json'" >&2
  exit 1
fi

sanitize() {
  local name=$1
  # Replace characters outside of [-_.a-zA-Z0-9] with underscores
  printf '%s' "${name//[^A-Za-z0-9._-]/_}"
}

for model in "${models[@]}"; do
  safe_model=$(sanitize "${model}")
  for input_path in "${inputs[@]}"; do
    filename=$(basename "${input_path}")
    year=${filename#aime}
    year=${year%.json}

    if [[ -z "${year}" ]]; then
      echo "Unable to extract year from '${filename}', skipping" >&2
      continue
    fi

    out_dir="data/aime_traces/${safe_model}"
    out_path="${out_dir}/traces_aime${year}.jsonl"

    echo "Generating traces for model '${model}' year '${year}'"
    python "${SCRIPT}" \
      --aime "${input_path}" \
      --out "${out_path}" \
      --model "${model}"
  done
done

echo "All trace generation jobs completed."
