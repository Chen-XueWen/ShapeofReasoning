#!/usr/bin/env python3
from __future__ import annotations

"""Combine per-target Stargazer HTML tables into a single matrix-style HTML table."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from bs4 import BeautifulSoup

try:  # Prefer the canonical feature ordering from the regression script
    from regression_scripts.analyze_tda_vs_graph import (
        GRAPH_TARGET_COLUMNS as DEFAULT_TARGETS,
        TDA_FEATURE_COLUMNS as DEFAULT_FEATURES,
    )
except Exception:  # pragma: no cover - fallback only used if import fails
    DEFAULT_TARGETS = [
        "avg_path_length",
        "diameter",
        "loop_count",
        "small_world_index",
        "avg_clustering",
    ]
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


def read_model_table(html_path: Path, feature_names: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """Parse a Stargazer HTML table and return coef/std pairs keyed by feature name."""
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
    base_dir: Path,
    targets: Iterable[str],
    features: Iterable[str],
    year_label: str,
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Aggregate coefficient tables for each target into nested dictionaries."""
    targets = list(targets)
    features = list(features)

    data: Dict[str, Dict[str, Tuple[str, str]]] = {
        feature: {} for feature in features + [INTERCEPT_LABEL]
    }

    for target in targets:
        html_path = base_dir / target / year_label / "ols_stargazer.html"
        table = read_model_table(html_path, features)

        for feature in data:
            coef, stderr = table.get(feature, ("", ""))
            data[feature][target] = (coef, stderr)

    return data


def render_html(
    features: List[str],
    targets: List[str],
    data: Dict[str, Dict[str, Tuple[str, str]]],
) -> str:
    """Render nested dictionary data into an HTML table."""
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
    header_cells = ["<th>TDA Feature</th>"] + [f"<th>{target}</th>" for target in targets]
    lines.append("<tr>" + "".join(header_cells) + "</tr>")

    for feature in features + [INTERCEPT_LABEL]:
        row_cells = [f"<td>{feature}</td>"]
        for target in targets:
            coef, stderr = data.get(feature, {}).get(target, ("", ""))
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
    features: List[str],
    targets: List[str],
    data: Dict[str, Dict[str, Tuple[str, str]]],
) -> None:
    """Write the combined data to a CSV file using plain text entries."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["TDA Feature", *targets])
        for feature in features + [INTERCEPT_LABEL]:
            row: List[str] = [feature]
            for target in targets:
                coef, stderr = data.get(feature, {}).get(target, ("", ""))
                if coef and stderr:
                    row.append(f"{coef} {stderr}")
                else:
                    row.append(coef or "")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-target Stargazer HTML outputs into a single table."
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Path to the tda_vs_graph analysis directory for a single model",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Dependent variables to include as table columns",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Ordered TDA feature names to include as rows",
    )
    parser.add_argument(
        "--year",
        default="combined",
        help="Year label to pull from each target directory (default: combined)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write the combined HTML table (default: <base_dir>/combined_table_<year>.html)",
    )

    args = parser.parse_args()

    base_dir: Path = args.base_dir
    if not base_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {base_dir}")

    targets = [str(t) for t in args.targets]
    features = [str(f) for f in args.features]

    data = build_matrix(base_dir, targets, features, args.year)
    html = render_html(features, targets, data)

    output_path = (
        args.output
        if args.output is not None
        else base_dir / f"combined_table_{args.year}.html"
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote combined table to {output_path}")

    csv_path = output_path.with_suffix(".csv")
    write_csv(csv_path, features, targets, data)
    print(f"Also wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
