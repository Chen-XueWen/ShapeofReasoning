#!/usr/bin/env bash
set -euo pipefail

models=(
  "qwen3:8b"
  "qwen3:30b"
  "deepseek-r1:8b"
  "deepseek-r1:32b"
  "gpt-oss:20b"
  "gpt-oss:120b"
  "qwen3:235b"
)

slugify() {
  # Replace characters unsafe for paths with underscores
  echo "$1" | tr ':/ ' '___'
}

for model in "${models[@]}"; do
  slug="$(slugify "$model")"
  echo "=== Running experiment for model: $model (slug: $slug) ==="

  traces="data/raw/traces_${slug}.jsonl"
  emb_dir="data/processed/${slug}/embeddings"
  feats="data/processed/${slug}/tda_features.jsonl"
  aligns="data/processed/${slug}/alignments.jsonl"
  analysis_dir="data/processed/analysis/${slug}"

  # 1) Generate reasoning traces (conditioned on provided answer)
  python scripts/generate_traces.py \
    --aime data/raw/aime2024.jsonl \
    --out "$traces" \
    --model "$model"

  ## 2) Embed steps
  #python scripts/embed_steps.py \
  #  --traces "$traces" \
  #  --outdir "$emb_dir"

  ## 3) Extract TDA features
  #python scripts/tda_extract.py \
  #  --emb-dir "$emb_dir" \
  #  --out "$feats"

  # 4) Align model steps to gold steps
  #python scripts/align_to_gold.py \
  #  --aime data/raw/aime2024.jsonl \
  #  --traces "$traces" \
  #  --out "$aligns"

  # 5) OLS analysis
  #python scripts/analyze_tda_vs_alignment.py \
  #  --align "$aligns" \
  #  --features "$feats" \
  #  --outdir "$analysis_dir"

  echo "=== Completed: results in $analysis_dir ==="
done

echo "All experiments completed."
