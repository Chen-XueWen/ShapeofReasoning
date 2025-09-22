#!/usr/bin/env python3
from __future__ import annotations

"""Combine Stargazer regression tables across models into a single matrix."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from bs4 import BeautifulSoup

try:
    from regression_scripts.analyze_tda_vs_alignment import FEATURE_COLUMNS as DEFAULT_FEATURES
except Exception:
    DEFAULT_FEATURES = [
        "H0_count",
        "H0_total_life",
        "H0_max_life",
        "H0_mean_life",
        "H0_entropy",
        "H0_skewness",
        "H0_max_birth",
        "H0_max_death",
        "H1_count",
        "H1_total_life",
        "H1_max_life",
        "H1_mean_life",
        "H1_entropy",
        "H1_skewness",
        "H1_max_birth",
        "H1_max_death",
        "H0_betti_peak",
        "H0_betti_location",
        "H0_betti_width",
        "H0_betti_centroid",
        "H0_betti_spread",
        "H0_betti_trend",
        "H1_betti_peak",
        "H1_betti_location",
        "H1_betti_width",
        "H1_betti_centroid",
        "H1_betti_spread",
        "H1_betti_trend",
        "H0_landscape_mean",
        "H0_landscape_max",
        "H0_landscape_area",
        "H1_landscape_mean",
        "H1_landscape_max",
        "H1_landscape_area",
    ]

INTERCEPT_LABEL = "const"


def discover_models(base_dir: Path, subpath: Path) -> List[Path]:
    paths: List[Path] = []
    for candidate in sorted(base_dir.iterdir()):
        target = candidate / subpath
        if target.exists():
            paths.append(candidate)
    if not paths:
        raise FileNotFoundError(
            f"No regression tables found under {base_dir}/*/{subpath}"
        )
    return paths


def read_model_table(html_path: Path, feature_names: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    if not html_path.exists():
        raise FileNotFoundError(f"Missing input table: {html_path}")

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No <table> found in {html_path}")

    feature_set = set(feature_names)
    feature_set.add(INTERCEPT_LABEL)

    results: Dict[str, Tuple[str, str]] = {}
    rows = table.find_all("tr")
    i = 0
    while i < len(rows):
        cells = rows[i].find_all("td")
        if len(cells) < 2:
            i += 1
            continue

        label = cells[0].get_text(strip=True)
        if label not in feature_set:
            i += 1
            continue

        coef = "".join(cells[1].stripped_strings)
        stderr = ""
        if i + 1 < len(rows):
            next_cells = rows[i + 1].find_all("td")
            if len(next_cells) >= 2 and next_cells[0].get_text(strip=True) == "":
                stderr = "".join(next_cells[1].stripped_strings)
                i += 1
        results[label] = (coef, stderr)
        i += 1

    return results


def build_matrix(
    models: Sequence[Path],
    features: Sequence[str],
    subpath: Path,
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    data: Dict[str, Dict[str, Tuple[str, str]]] = {
        feature: {} for feature in list(features) + [INTERCEPT_LABEL]
    }

    for model_dir in models:
        model_name = model_dir.name
        html_path = model_dir / subpath
        table = read_model_table(html_path, features)
        for feature in data:
            data[feature][model_name] = table.get(feature, ("", ""))

    return data


def render_html(
    features: Sequence[str],
    models: Sequence[str],
    data: Dict[str, Dict[str, Tuple[str, str]]],
) -> str:
    style = (
        "<style>table {border-collapse: collapse; font-family: Arial, sans-serif;}"
        " th, td {border: 1px solid #ccc; padding: 6px 10px;}"
        " th {background-color: #f5f5f5;}"
        " td {text-align: center;}"
        " td:first-child {text-align: left; font-weight: 600;}"
        " .stderr {display: block; color: #555; font-size: 0.85em;}"
        "</style>"
    )

    lines = [style, "<table>"]
    header = ["<th>Feature</th>"] + [f"<th>{name}</th>" for name in models]
    lines.append("<tr>" + "".join(header) + "</tr>")

    for feature in list(features) + [INTERCEPT_LABEL]:
        row_cells = [f"<td>{feature}</td>"]
        for model in models:
            coef, stderr = data.get(feature, {}).get(model, ("", ""))
            if coef and stderr:
                cell = f"{coef}<br><span class=\"stderr\">{stderr}</span>"
            else:
                cell = coef or ""
            row_cells.append(f"<td>{cell}</td>")
        lines.append("<tr>" + "".join(row_cells) + "</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def write_csv(
    path: Path,
    features: Sequence[str],
    models: Sequence[str],
    data: Dict[str, Dict[str, Tuple[str, str]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Feature", *models])
        for feature in list(features) + [INTERCEPT_LABEL]:
            row: List[str] = [feature]
            for model in models:
                coef, stderr = data.get(feature, {}).get(model, ("", ""))
                if coef and stderr:
                    row.append(f"{coef} {stderr}")
                else:
                    row.append(coef or "")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Stargazer regression tables across models into one output table",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=Path("analysis"),
        type=Path,
        help="Directory containing per-model analysis folders (default: analysis)",
    )
    parser.add_argument(
        "--subpath",
        default="tda_vs_alignment/combined/ols_stargazer.html",
        help="Relative path from each model directory to the Stargazer HTML file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names or paths to include (defaults to discovery under base_dir)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Ordered feature names to include as rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination for the combined HTML table (default inferred from subpath)",
    )

    args = parser.parse_args()

    base_dir: Path = args.base_dir
    subpath = Path(args.subpath)

    if args.models:
        model_dirs: List[Path] = []
        for entry in args.models:
            path = Path(entry)
            if not path.is_absolute():
                path = base_dir / path
            target = path / subpath
            if path.is_dir() and target.exists():
                model_dirs.append(path)
            else:
                raise FileNotFoundError(
                    f"Could not locate {subpath} under provided model path: {path}"
                )
    else:
        model_dirs = discover_models(base_dir, subpath)

    if not model_dirs:
        raise ValueError("No models provided or discovered")

    features = list(args.features)
    models = [path.name for path in model_dirs]

    data = build_matrix(model_dirs, features, subpath)
    html = render_html(features, models, data)

    if args.output is not None:
        output_path = args.output
    else:
        prefix = subpath.parts[0] if subpath.parts else "combined"
        output_path = base_dir / f"{prefix}_combined_models.html"

    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote combined table to {output_path}")

    csv_path = output_path.with_suffix(".csv")
    write_csv(csv_path, features, models, data)
    print(f"Also wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
