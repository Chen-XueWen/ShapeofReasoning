#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from tda_reasoning.tda.vr import vr_diagrams, vr_features
from tda_reasoning.tda.betti import betti_features


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute TDA features from diagrams")
    ap.add_argument("--diag-dir", default="data/processed/diagrams", help="Diagrams dir")
    ap.add_argument("--out", default="data/processed/tda_features.jsonl", help="Output JSONL")
    ap.add_argument("--maxdim", type=int, default=1, help="Max homology dimension")
    args = ap.parse_args()

    ensure_dir(args.out)
    out_f = open(args.out, "w", encoding="utf-8")
    diag_dir = Path(args.diag_dir)
    n = 0
    for npz_path in tqdm(sorted(diag_dir.glob("*.npz")), desc="Computing TDA", unit="file"):
        pid = npz_path.stem
        dgms = load_npz(npz_path)
        feats = vr_features(dgms)
        feats.update(betti_features(dgms))
        row = {"id": pid, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in feats.items()}}
        out_f.write(json.dumps(row) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote TDA features for {n} items to {args.out}")


if __name__ == "__main__":
    main()
