#!/usr/bin/env bash

set -euo pipefail

TDA_SCRIPT="./feature_scripts/tda_extract.py"

if [[ ! -f "${TDA_SCRIPT}" ]]; then
  echo "TDA extraction script '${TDA_SCRIPT}' not found" >&2
  exit 1
fi

MODELS=(
  #"deepseek-r1:32b"
  #"qwen3:32b"
  #"gpt-oss:20b"
  #"gpt-oss:120b"
  #"deepseek-r1:8b"
  "deepseek-r1:70b"
  #"qwen3:8b"
  #"qwen3:235b"
)

SPLITS=(
  "trace"
  "gold"
)

sanitize() {
  local name=$1
  printf '%s' "${name//[^A-Za-z0-9._-]/_}"
}

for model in "${MODELS[@]}"; do
  safe_model=$(sanitize "${model}")

  for split in "${SPLITS[@]}"; do
    EMB_DIR="data/aime_embed/${safe_model}/${split}"
    if [[ ! -d "${EMB_DIR}" ]]; then
      echo "Embedding directory '${EMB_DIR}' not found" >&2
      exit 1
    fi

    echo "Running TDA extraction for '${model}' (${split})"
    python3 "${TDA_SCRIPT}" \
      --model "${safe_model}" \
      --split "${split}" \
      "$@"
  done
done

echo "TDA extraction completed for all models."
