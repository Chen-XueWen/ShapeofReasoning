#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from tda_reasoning.tda.vr import vr_diagrams

def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
    
def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    ensure_dir(path)
    np.savez_compressed(path, allow_pickle=True, **arrays)
    
def main() -> None:
    ap = argparse.ArgumentParser(description="Compute TDA features from embeddings")
    ap.add_argument("--emb-dir", default="data/processed/embeddings", help="Embeddings dir")
    ap.add_argument("--out", default="data/processed/diagrams/", help="Output dir")
    ap.add_argument("--metric", default="cosine", help="Distance metric for VR")
    ap.add_argument("--maxdim", type=int, default=1, help="Max homology dimension")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    n = 0
    for npz_path in tqdm(sorted(emb_dir.glob("*.npz")), desc="Computing TDA", unit="file"):
        pid = npz_path.stem
        arrays = load_npz(npz_path)
        X = arrays.get("X")
        if X is None or X.shape[0] < 2:
            continue
        dgms = vr_diagrams(X, metric=args.metric, maxdim=args.maxdim)
        out_path = Path(args.out) / f"{pid}.npz"
        save_npz(out_path, **{f"H{dim}": dgm for dim, dgm in dgms.items()})
        n += 1
    print(f"Wrote TDA features for {n} items to {args.out}")


if __name__ == "__main__":
    main()
