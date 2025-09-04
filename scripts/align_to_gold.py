#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from tda_reasoning.embedding.segment import segment_steps
from tda_reasoning.embedding.embedder import EmbeddingConfig, SentenceTransformerEmbedder
from tda_reasoning.eval.align import align_steps


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Align model steps to gold solution steps")
    ap.add_argument("--aime", default="data/raw/aime2024.jsonl", help="AIME JSONL with gold text")
    ap.add_argument("--traces", default="data/raw/traces_aime2024.jsonl", help="Model traces JSONL")
    ap.add_argument("--out", default="data/processed/alignments.jsonl", help="Output JSONL")
    ap.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model for alignment",
    )
    args = ap.parse_args()

    ensure_dir(args.out)

    gold_by_id: dict[str, list[str]] = {}
    for ex in read_jsonl(args.aime):
        gold_steps = ex.get("gold_steps")
        if not gold_steps and ex.get("solution"):
            gold_steps = segment_steps(ex["solution"])  # best-effort
        gold_by_id[str(ex.get("id"))] = gold_steps or []

    embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name=args.model_name))

    out_f = open(args.out, "w", encoding="utf-8")
    n = 0
    for tr in tqdm(read_jsonl(args.traces), desc="Aligning traces", unit="trace"):
        pid = str(tr.get("id"))
        trace_steps = segment_steps(tr.get("trace", ""))
        gold_steps = gold_by_id.get(pid, [])
        if not trace_steps or not gold_steps:
            continue
        X_model = embedder.encode(trace_steps)
        X_gold = embedder.encode(gold_steps)
        align = align_steps(X_model, X_gold)
        # Ensure JSON-serializable types (convert NumPy scalars to Python types)
        row = {
            "id": pid,
            "indices": [(int(i), int(j)) for i, j in align.indices],
            "score": float(align.score),
            "coverage": float(align.coverage),
        }
        out_f.write(json.dumps(row) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote {n} alignments to {args.out}")


if __name__ == "__main__":
    main()
