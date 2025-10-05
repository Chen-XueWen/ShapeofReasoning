#!/usr/bin/env python3
from __future__ import annotations

"""Build a unified regression dataset CSV from model JSONL inputs."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

DEFAULT_MODELS: Tuple[str, ...] = (
    "deepseek-r1_32b",
    "gpt-oss_120b",
    "gpt-oss_20b",
    "qwen3_32b",
    "qwen3_8b",
    "deepseek-r1_8b",
    "deepseek-r1_70b",
)

TDA_SUBPATH = Path("data/aime_tda/trace")
GRAPH_SUBPATH = Path("data/aime_graph/trace")
ALIGN_SUBPATH = Path("data/aime_align_dp")
DEFAULT_OUTPUT = Path("data/aime_regression_dataset_cleaned.csv")


EXCLUDED_FIELDS = {
    "H0_betti_trend",
    "H1_betti_trend",
    "H0_betti_peak",
    "H0_betti_location",
    "H0_max_birth",
    "H0_max_death",
}


@dataclass(frozen=True)
class DatasetPaths:
    tda: Path
    graph: Path
    align: Path


def load_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
            if not isinstance(payload, Mapping):
                raise TypeError(f"Expected JSON object on {path}:{line_no}")
            yield payload


def parse_problem_id(problem_id: str) -> Tuple[str, str, str]:
    parts = problem_id.split("-")
    if len(parts) >= 3:
        year, exam, problem = parts[0], parts[1], "-".join(parts[2:])
    elif len(parts) == 2:
        year, exam, problem = parts[0], parts[1], ""
    else:
        year, exam, problem = problem_id, "", ""
    return year, exam, problem


def normalise_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def collect_records(
    models: Sequence[str],
    paths: DatasetPaths,
) -> Dict[Tuple[str, str], MutableMapping[str, object]]:
    records: Dict[Tuple[str, str], MutableMapping[str, object]] = {}
    coverage_by_source: Dict[str, set[Tuple[str, str]]] = defaultdict(set)

    def ensure_record(model: str, problem_id: str) -> MutableMapping[str, object]:
        key = (model, problem_id)
        if key not in records:
            year, exam, problem = parse_problem_id(problem_id)
            records[key] = {
                "model": model,
                "id": problem_id,
                "year": year,
                "exam": exam,
                "problem": problem,
            }
        return records[key]

    for model in models:
        tda_path = paths.tda / f"{model}.jsonl"
        for entry in load_jsonl(tda_path):
            problem_id = str(entry.get("id"))
            record = ensure_record(model, problem_id)
            for key, value in entry.items():
                if key == "id":
                    continue
                record[f"tda_{key}"] = normalise_value(value)
            coverage_by_source["tda"].add((model, problem_id))

        graph_path = paths.graph / f"{model}.jsonl"
        for entry in load_jsonl(graph_path):
            problem_id = str(entry.get("id"))
            record = ensure_record(model, problem_id)
            for key, value in entry.items():
                if key == "id":
                    continue
                record[f"graph_{key}"] = normalise_value(value)
            coverage_by_source["graph"].add((model, problem_id))

        align_dir = paths.align / model
        if not align_dir.exists():
            raise FileNotFoundError(f"Missing alignment directory: {align_dir}")
        for jsonl_path in sorted(align_dir.glob("*.jsonl")):
            for entry in load_jsonl(jsonl_path):
                problem_id = str(entry.get("id"))
                record = ensure_record(model, problem_id)
                for key, value in entry.items():
                    if key == "id":
                        continue
                    record[f"align_{key}"] = normalise_value(value)
                coverage_by_source["align"].add((model, problem_id))

    expected_keys = coverage_by_source.get("tda", set())
    missing_messages: List[str] = []
    for source, keys in coverage_by_source.items():
        missing = expected_keys - keys
        if missing:
            missing_messages.append(
                f"{source} missing {len(missing)} records compared to TDA"
            )
    if missing_messages:
        raise ValueError("; ".join(missing_messages))

    return records


def write_csv(
    records: Dict[Tuple[str, str], MutableMapping[str, object]],
    output_path: Path,
) -> int:
    rows = list(records.values())
    if not rows:
        raise ValueError("No records collected; nothing to write")

    base_fields = ["model", "id", "year", "exam", "problem"]
    feature_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_fields
            and key not in EXCLUDED_FIELDS
        }
    )
    fieldnames = base_fields + feature_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered_row = {
                key: value
                for key, value in row.items()
                if key not in EXCLUDED_FIELDS
            }
            writer.writerow(filtered_row)

    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Subset of model names to include (default: all known models)",
    )
    parser.add_argument(
        "--tda-dir",
        type=Path,
        default=TDA_SUBPATH,
        help="Directory containing TDA trace JSONL files",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=GRAPH_SUBPATH,
        help="Directory containing graph trace JSONL files",
    )
    parser.add_argument(
        "--align-dir",
        type=Path,
        default=ALIGN_SUBPATH,
        help="Directory containing alignment JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path",
    )
    args = parser.parse_args(argv)

    dataset_paths = DatasetPaths(
        tda=args.tda_dir,
        graph=args.graph_dir,
        align=args.align_dir,
    )

    records = collect_records(args.models, dataset_paths)
    row_count = write_csv(records, args.output)
    print(f"Wrote {row_count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
