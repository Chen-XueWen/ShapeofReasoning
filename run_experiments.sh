#!/usr/bin/env bash
# set -euo pipefail

models=(
  "gemma3:27b"
  # "qwen3:8b"
  # "deepseek-r1:8b"
  # "qwen3:30b,1"
  # "qwen3:30b,0"
  # "deepseek-r1:32b,1"
  # "deepseek-r1:32b,0"
  # "gpt-oss:20b,low"
  # "gpt-oss:20b,high"
  # "gpt-oss:120b,low"
  # "gpt-oss:120b,high"
  # "qwen3:235b"
)

slugify() {
  # Replace characters unsafe for paths with underscores
  echo "$1" | tr ':/, ' '___'
}

for setup in "${models[@]}"; do
  model="${setup%%,*}"
  think_level="${setup#*,}"
  slug="$(slugify "$setup")"
  echo "=== Running experiment for setup: $setup (slug: $slug) ==="

  traces="data/raw/traces_${slug}.jsonl"
  emb_dir="data/processed/${slug}/embeddings"
  diag_dir="data/processed/${slug}/diagrams"
  feats="data/processed/${slug}/tda_features.jsonl"
  aligns="data/processed/${slug}/alignments.jsonl"
  analysis_dir="data/processed/analysis/${slug}"

  # # 1) Generate reasoning traces (conditioned on provided answer)
  # python scripts/generate_traces_no_ans.py \
  #   --aime data/raw/aime2024.jsonl \
  #   --out "$traces" \
  #   --model "$model" \
  #   --think-level "$think_level" \
  #   --max_tokens "-1"

  # # 2) Embed steps
  # python scripts/embed_steps.py \
  #  --traces "$traces" \
  #  --outdir "$emb_dir"

  # 3) Generate Persistent Homology diagrams
  python scripts/create_diags.py \
    --emb-dir "$emb_dir" \
    --out "$diag_dir" \
    --metric "cosine" \
    --maxdim 1

  # 4) Extract TDA features
  python scripts/extract_features.py \
   --diag-dir "$diag_dir" \
   --out "$feats"

  # 5) Align model steps to gold steps
  python scripts/align_to_gold.py \
   --aime data/raw/aime2024.jsonl \
   --traces "$traces" \
   --out "$aligns"

  # 5) OLS analysis
  python scripts/analyze_tda_vs_alignment.py \
   --align "$aligns" \
   --features "$feats" \
   --outdir "$analysis_dir"

  echo "=== Completed: results in $analysis_dir ==="
done

echo "All experiments completed."
