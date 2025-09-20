#!/usr/bin/env bash
# set -euo pipefail

slugify() {
  # Replace characters unsafe for paths with underscores
  echo "$1" | tr ':/, ' '___'
}

model=$1
for i in $(seq 1 $2); do
  slug="$(slugify "${model},${i}")"
  echo "=== Running experiment for setup: $model, $i (slug: $slug) ==="

  traces="past_data/new_segments/raw/traces_${slug}.jsonl"
  emb_dir="past_data/new_segments/processed/${slug}/embeddings"
  diag_dir="past_data/new_segments/processed/${slug}/diagrams"
  feats="past_data/new_segments/processed/${slug}/tda_features.jsonl"
  aligns="past_data/new_segments/processed/${slug}/alignments.jsonl"
  analysis_dir="past_data/new_segments/processed/analysis/${slug}"

  # 1) Generate reasoning traces (conditioned on provided answer)
  # python scripts/generate_traces_no_ans.py \
  #   --aime data/raw/aime2024.jsonl \
  #   --out "$traces" \
  #   --model "$model" \
  #   --max_tokens "-1"

  # # 2) Embed steps
  # python scripts/embed_steps.py \
  #  --traces "$traces" \
  #  --outdir "$emb_dir"

  # # 3) Generate Persistent Homology diagrams
  # python scripts/create_diags.py \
  #   --emb-dir "$emb_dir" \
  #   --out "$diag_dir" \
  #   --metric "cosine" \
  #   --maxdim 1

  # # 4) Extract TDA features
  # python scripts/extract_features.py \
  #  --diag-dir "$diag_dir" \
  #  --out "$feats"

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
