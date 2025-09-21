#!/usr/bin/env python3

import sys
from pathlib import Path
# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
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
    ap.add_argument("--traces", default="data/aime_traces/deepseek-r1_32b/traces_aime2025.jsonl", help="Input JSONL")
    ap.add_argument("--outdir", default="data/aime_embed/deepseek-r1_32b/", help="Output dir")
    ap.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-transformers embedding model",
    )
    ap.add_argument("--device", default="cuda:7", help="Device override (e.g., cuda or cpu)")
    args = ap.parse_args()

    embedder = SentenceTransformerEmbedder(EmbeddingConfig(model_name=args.model_name, device=args.device))
    n = 0
    for row in tqdm(read_jsonl(args.traces), desc="Embedding steps", unit="trace"):
        pid = row.get("id")
        trace = row.get("trace")
        steps: list[str] = segment_steps(trace)
        X = embedder.encode(steps)
        out_path = Path(args.outdir) / f"{pid}.npz"
        save_npz(out_path, X=X)
        n += 1
    print(f"Embedded steps for {n} traces into {args.outdir}")


if __name__ == "__main__":
    main()
