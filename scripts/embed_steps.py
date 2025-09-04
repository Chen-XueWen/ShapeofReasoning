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
import numpy as np
from tqdm import tqdm

from tda_reasoning.embedding.segment import segment_steps
from tda_reasoning.embedding.embedder import EmbeddingConfig, SentenceTransformerEmbedder


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    ensure_dir(path)
    np.savez_compressed(path, allow_pickle=True, **arrays)


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment traces and compute embeddings")
    ap.add_argument("--traces", default="data/raw/traces_aime2024.jsonl", help="Input JSONL")
    ap.add_argument("--outdir", default="data/processed/embeddings", help="Output dir")
    ap.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model",
    )
    ap.add_argument("--device", default="cpu", help="Device override (e.g., cuda or cpu)")
    args = ap.parse_args()

    embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name=args.model_name, device=args.device))
    n = 0
    for row in tqdm(read_jsonl(args.traces), desc="Embedding steps", unit="trace"):
        pid = row.get("id")
        trace = row.get("trace", "")
        steps: list[str] = segment_steps(trace)
        if not steps:
            steps = [trace] if trace else []
        if not steps:
            continue
        X = embedder.encode(steps)
        out_path = Path(args.outdir) / f"{pid}.npz"
        save_npz(out_path, X=X)
        n += 1
    print(f"Embedded steps for {n} traces into {args.outdir}")


if __name__ == "__main__":
    main()
