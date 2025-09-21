#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import IO, Tuple
import sys

# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from tda_reasoning.tda.vr import (
    landscape_features,
    vr_diagrams,
    vr_features,
)
from tda_reasoning.tda.betti import betti_features


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    ensure_dir(path)
    np.savez_compressed(path, allow_pickle=True, **arrays)


def compute_diagrams(
    embeddings: np.ndarray,
    metric: str,
    maxdim: int,
) -> dict[str, np.ndarray]:
    diagrams = vr_diagrams(embeddings, metric=metric, maxdim=maxdim)
    return {f"H{dim}": dgm for dim, dgm in diagrams.items()}


def compute_feature_row(
    pid: str,
    diagrams: dict[str, np.ndarray],
) -> dict[str, object]:
    feats: dict[str, object] = {}
    feats.update(vr_features(diagrams))
    feats.update(betti_features(diagrams))
    feats.update(landscape_features(diagrams))
    serialisable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in feats.items()
    }
    return {"id": pid, **serialisable}


def embeddings_to_diagrams_and_features(
    emb_dir: Path,
    diagram_out_dir: Path,
    feature_writer: IO[str],
    metric: str,
    maxdim: int,
    reuse_diagrams: bool = False,
) -> Tuple[int, int, int]:
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory '{emb_dir}' not found")

    diagram_out_dir.mkdir(parents=True, exist_ok=True)

    diag_written = 0
    diag_reused = 0
    feat_count = 0

    for npz_path in tqdm(sorted(emb_dir.glob("*.npz")), desc="Computing TDA", unit="file"):
        pid = npz_path.stem
        diag_path = diagram_out_dir / f"{pid}.npz"

        diagrams: dict[str, np.ndarray]
        if reuse_diagrams and diag_path.exists():
            diagrams = load_npz(diag_path)
            diag_reused += 1
        else:
            arrays = load_npz(npz_path)
            X = arrays.get("X")
            if X is None or X.shape[0] < 2:
                continue
            diagrams = compute_diagrams(X, metric=metric, maxdim=maxdim)
            save_npz(diag_path, **diagrams)
            diag_written += 1

        row = compute_feature_row(pid, diagrams)
        feature_writer.write(json.dumps(row) + "\n")
        feat_count += 1

    return diag_written, diag_reused, feat_count


def diagrams_dir_to_features(
    diag_dir: Path,
    feature_writer: IO[str],
    desc: str = "Computing TDA features",
) -> int:
    if not diag_dir.exists():
        raise FileNotFoundError(f"Diagram directory '{diag_dir}' not found")

    feature_count = 0
    for npz_path in tqdm(sorted(diag_dir.glob("*.npz")), desc=desc, unit="file"):
        pid = npz_path.stem
        diagrams = load_npz(npz_path)
        row = compute_feature_row(pid, diagrams)
        feature_writer.write(json.dumps(row) + "\n")
        feature_count += 1
    return feature_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute persistent diagrams and TDA features")
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
        "--diag-root",
        default="data/aime_tda_diags",
        help="Root directory where persistent diagrams will be written",
    )
    ap.add_argument(
        "--features-root",
        default="data/aime_tda",
        help="Root directory where feature JSONL files will be written",
    )
    ap.add_argument(
        "--emb-dir",
        default=None,
        help="Override embeddings directory (bypasses root/model/split layout)",
    )
    ap.add_argument(
        "--diag-dir",
        default=None,
        help="Override diagram directory (bypasses root/model/split layout)",
    )
    ap.add_argument(
        "--features-out",
        default=None,
        help="Override features JSONL path (bypasses root/model/split layout)",
    )
    ap.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric for Vietoris-Rips diagrams",
    )
    ap.add_argument(
        "--maxdim",
        type=int,
        default=1,
        help="Maximum homology dimension",
    )
    ap.add_argument(
        "--reuse-diagrams",
        action="store_true",
        help="Reuse existing diagram files if present instead of recomputing",
    )
    args = ap.parse_args()

    model_slug = args.model
    split = args.split

    emb_dir = Path(args.emb_dir) if args.emb_dir else Path(args.embed_root) / model_slug / split
    diag_dir = Path(args.diag_dir) if args.diag_dir else Path(args.diag_root) / model_slug / split
    if args.features_out:
        features_out = Path(args.features_out)
    else:
        features_out = Path(args.features_root) / split / f"{model_slug}.jsonl"

    ensure_dir(features_out)

    with open(features_out, "w", encoding="utf-8") as feature_writer:
        diag_written, diag_reused, feat_count = embeddings_to_diagrams_and_features(
            emb_dir=emb_dir,
            diagram_out_dir=diag_dir,
            feature_writer=feature_writer,
            metric=args.metric,
            maxdim=args.maxdim,
            reuse_diagrams=args.reuse_diagrams,
        )

    total_diagrams = diag_written + diag_reused
    if args.reuse_diagrams and diag_reused > 0:
        print(
            f"Processed {split} diagrams for {total_diagrams} items "
            f"({diag_written} recomputed, {diag_reused} reused) in {diag_dir}"
        )
    else:
        print(f"Wrote {split} diagrams for {diag_written} items to {diag_dir}")

    print(f"Wrote {split} features for {feat_count} items to {features_out}")


if __name__ == "__main__":
    main()
