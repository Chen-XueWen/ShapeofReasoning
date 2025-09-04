#!/usr/bin/env python3

import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tda_reasoning.tda.features import compute_diagrams, assemble_feature_vector


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute TDA features from embeddings")
    ap.add_argument("--emb-dir", default="data/processed/embeddings", help="Embeddings dir")
    ap.add_argument("--out", default="data/processed/tda_features.jsonl", help="Output JSONL")
    ap.add_argument("--metric", default="cosine", help="Distance metric for VR")
    ap.add_argument("--maxdim", type=int, default=1, help="Max homology dimension")
    args = ap.parse_args()

    ensure_dir(args.out)
    out_f = open(args.out, "w", encoding="utf-8")
    emb_dir = Path(args.emb_dir)
    n = 0
    for npz_path in tqdm(sorted(emb_dir.glob("*.npz")), desc="Computing TDA", unit="file"):
        pid = npz_path.stem
        arrays = load_npz(npz_path)
        X = arrays.get("X")
        if X is None or X.shape[0] < 2:
            continue
        dgms = compute_diagrams(X, metric=args.metric, maxdim=args.maxdim)
        feats = assemble_feature_vector(dgms)
        row = {"id": pid, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in feats.items()}}
        out_f.write(json.dumps(row) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote TDA features for {n} items to {args.out}")


if __name__ == "__main__":
    main()
