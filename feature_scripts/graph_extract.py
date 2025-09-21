import argparse
import json
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tda_reasoning.eval.graph import analyze_graph_simple, build_graph


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def process_embeddings_dir(
    emb_dir: Path,
    out_path: Path,
) -> int:
    ensure_dir(out_path)
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for npz_path in tqdm(sorted(emb_dir.glob("*.npz")), desc=f"Graph features ({emb_dir.name})", unit="file"):
            pid = npz_path.stem
            arrays = load_npz(npz_path)
            X = arrays.get("X")
            if X is None or X.shape[0] < 2:
                continue
            path, distances = build_graph(X)
            (
                has_loop,
                loop_count,
                diameter,
                avg_clustering,
                avg_path_length,
                small_world_index,
            ) = analyze_graph_simple(path, distances)
            row = {
                "id": pid,
                "has_loop": has_loop,
                "loop_count": loop_count,
                "diameter": diameter,
                "avg_clustering": avg_clustering,
                "avg_path_length": avg_path_length,
                "small_world_index": small_world_index,
            }
            out_f.write(json.dumps(row) + "\n")
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute graph features from embeddings")
    ap.add_argument(
        "--model",
        default="gpt-oss_20b",
        help="Model slug used in directory layout (e.g. deepseek-r1_32b)",
    )
    ap.add_argument(
        "--split",
        choices=("trace", "gold"),
        default="trace",
        help="Which embedding split to process",
    )
    ap.add_argument(
        "--embed-root",
        default="data/aime_embed",
        help="Root directory containing per-model embedding folders",
    )
    ap.add_argument(
        "--features-root",
        default="data/aime_graph",
        help="Root directory where graph feature JSONL files will be written",
    )
    ap.add_argument(
        "--emb-dir",
        default=None,
        help="Override embeddings directory (bypasses root/model/split layout)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Override output JSONL path (bypasses root/model/split layout)",
    )
    args = ap.parse_args()

    model_slug = args.model
    split = args.split

    emb_dir = Path(args.emb_dir) if args.emb_dir else Path(args.embed_root) / model_slug / split
    features_out = Path(args.out) if args.out else Path(args.features_root) / split / f"{model_slug}.jsonl"

    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory '{emb_dir}' not found")

    count = process_embeddings_dir(emb_dir, features_out)
    print(f"Wrote {split} graph features for {count} items to {features_out}")

if __name__ == "__main__":
    main()
