import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm


# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tda_reasoning.eval.graph import analyze_graph_simple, build_graph


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}

def main() -> None:
    ap = argparse.ArgumentParser(description="Align model steps to gold solution steps")
    ap.add_argument("--aime", default="data/raw/aime2024.jsonl", help="AIME JSONL with gold text")
    ap.add_argument("--embeddings", default="data/processed/gpt-oss_20b_1/embeddings", help="Embeddings dir")
    ap.add_argument("--out", default="data/processed/graph_analysis.jsonl", help="Output JSONL")
    args = ap.parse_args()

    out_f = open(args.out, "w", encoding="utf-8")
    n = 0
    for npz_path in tqdm(sorted(Path(args.embeddings).glob("*.npz")), desc="Computing graph properties", unit="file"):
        pid = npz_path.stem
        arrays = load_npz(npz_path)
        X = arrays.get("X")
        if X is None or X.shape[0] < 2:
            continue
        path, distances = build_graph(X)
        has_loop, loop_count, diameter, avg_clustering, avg_path_length, small_world_index = analyze_graph_simple(path, distances)
        row = {"id": pid, "has_loop": has_loop, "loop_count": loop_count, "diameter": diameter, "avg_clustering": avg_clustering, "avg_path_length": avg_path_length, "small_world_index": small_world_index}
        out_f.write(json.dumps(row) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote graph analysis for {n} items to {args.out}")

if __name__ == "__main__":
    main()

