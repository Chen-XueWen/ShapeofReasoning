#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

Diagram = Dict[str, np.ndarray]

def load_diagram(path: Path) -> Diagram:
    arrays: Diagram = {}
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            arrays[key] = data[key]
    return arrays

def load_diagrams(split_dir: Path) -> List[Diagram]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Diagram directory '{split_dir}' not found")
    diagrams: List[Diagram] = []
    for npz_path in sorted(split_dir.glob("*.npz")):
        diagrams.append(load_diagram(npz_path))
    if not diagrams:
        raise ValueError(f"No .npz diagrams found in '{split_dir}'")
    return diagrams

def compute_grid(diagrams: Iterable[np.ndarray], grid_size: int) -> np.ndarray:
    births: List[float] = []
    deaths: List[float] = []
    for dgm in diagrams:
        if dgm.size == 0:
            continue
        births.append(float(np.min(dgm[:, 0])))
        finite = dgm[np.isfinite(dgm[:, 1])]
        if finite.size:
            deaths.append(float(np.max(finite[:, 1])))
    if not births or not deaths:
        raise ValueError("Could not determine filtration range (no finite intervals)")
    t_min = min(births)
    t_max = max(deaths)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"Invalid filtration bounds: min={t_min}, max={t_max}")
    return np.linspace(t_min, t_max, grid_size)

def betti_counts_over_grid(dgm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros_like(grid)
    births = dgm[:, 0][:, None]
    deaths = dgm[:, 1][:, None]
    alive = (births <= grid) & (grid < deaths)
    return alive.sum(axis=0).astype(float)

def landscape_level_over_grid(dgm: np.ndarray, grid: np.ndarray, level: int) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros_like(grid)
    finite_mask = np.isfinite(dgm[:, 1])
    if not finite_mask.any():
        return np.zeros_like(grid)
    births = dgm[finite_mask, 0][:, None]
    deaths = dgm[finite_mask, 1][:, None]
    left = grid - births
    right = deaths - grid
    heights = np.minimum(left, right)
    heights = np.where((left > 0) & (right > 0), heights, 0.0)
    if heights.shape[0] < level:
        return np.zeros_like(grid)
    heights.sort(axis=0)
    return heights[-level, :]

def aggregate_curves(
    diagrams: List[Diagram],
    dim_key: str,
    grid_size: int,
    landscape_level: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dgms = [d[dim_key] for d in diagrams if dim_key in d and d[dim_key].ndim == 2]
    if not dgms:
        raise ValueError(f"No diagrams for key '{dim_key}'")
    grid = compute_grid(dgms, grid_size)
    betti_curves = np.vstack([betti_counts_over_grid(dgm, grid) for dgm in dgms])
    landscape_curves = np.vstack([landscape_level_over_grid(dgm, grid, landscape_level) for dgm in dgms])
    betti_mean = betti_curves.mean(axis=0)
    betti_std = betti_curves.std(axis=0)
    landscape_mean = landscape_curves.mean(axis=0)
    landscape_std = landscape_curves.std(axis=0)
    return grid, betti_mean, betti_std, landscape_mean, landscape_std

def plot_curves(
    splits: Dict[str, List[Diagram]],
    dims: Iterable[str],
    grid_size: int,
    landscape_level: int,
    output_path: Path,
    show: bool,
) -> None:
    rows = len(splits)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows), sharex=False)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    colors = {"H0": "tab:blue", "H1": "tab:orange"}
    for row_idx, (split, diagrams) in enumerate(splits.items()):
        betti_ax = axes[row_idx, 0]
        landscape_ax = axes[row_idx, 1]
        betti_ax.set_title(f"{split} Betti curves")
        landscape_ax.set_title(f"{split} landscapes (level {landscape_level})")
        for dim_key in dims:
            try:
                grid, betti_mean, betti_std, landscape_mean, landscape_std = aggregate_curves(
                    diagrams, dim_key, grid_size, landscape_level
                )
            except ValueError:
                continue
            color = colors.get(dim_key, None)
            betti_ax.plot(grid, betti_mean, label=dim_key, color=color)
            betti_ax.fill_between(grid, betti_mean - betti_std, betti_mean + betti_std, color=color, alpha=0.2)
            landscape_ax.plot(grid, landscape_mean, label=dim_key, color=color)
            landscape_ax.fill_between(
                grid,
                landscape_mean - landscape_std,
                landscape_mean + landscape_std,
                color=color,
                alpha=0.2,
            )
        betti_ax.set_xlabel("Filtration value")
        betti_ax.set_ylabel("Betti")
        betti_ax.legend()
        landscape_ax.set_xlabel("Filtration value")
        landscape_ax.set_ylabel("Landscape height")
        landscape_ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot Betti curves and persistence landscapes from VR diagrams")
    ap.add_argument(
        "--diag-root",
        default="data/aime_tda_diags",
        help="Root directory containing per-model diagram folders",
    )
    ap.add_argument(
        "--model",
        default="deepseek-r1_32b",
        help="Model slug used inside the diagram root",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["gold", "trace"],
        help="Diagram subdirectories to include",
    )
    ap.add_argument(
        "--grid-size",
        type=int,
        default=400,
        help="Number of evaluation points for curves",
    )
    ap.add_argument(
        "--landscape-level",
        type=int,
        default=1,
        help="Persistence landscape level to plot (1 = first landscape)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Path to save the output figure (defaults to plots/<model>_tda_profiles.png)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    diag_root = Path(args.diag_root)
    model_dir = diag_root / args.model
    splits: Dict[str, List[Diagram]] = {}
    for split in args.splits:
        split_dir = model_dir / split
        splits[split] = load_diagrams(split_dir)
    output_path = Path(args.output) if args.output else Path("plots") / f"{args.model}_tda_profiles.png"
    plot_curves(
        splits=splits,
        dims=("H0", "H1"),
        grid_size=args.grid_size,
        landscape_level=args.landscape_level,
        output_path=output_path,
        show=args.show,
    )
    print(f"Saved Betti and landscape plots to {output_path}")


if __name__ == "__main__":
    main()
