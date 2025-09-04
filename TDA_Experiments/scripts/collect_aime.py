#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tda_reasoning.data.loader import (
    load_dataset_generic,
    load_dataset_via_hub_parquet,
    normalize_aime_examples,
    to_jsonl_rows,
    save_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect AIME dataset from Hugging Face")
    ap.add_argument("--dataset", default="Maxwell-Jia/AIME_2024", help="HF dataset name")
    ap.add_argument("--split", default="train", help="Split name (if using datasets lib)")
    ap.add_argument(
        "--field-map",
        default=None,
        help="JSON string mapping fields: statement, solution, answer, year, index, steps",
    )
    ap.add_argument(
        "--out",
        default="data/raw/aime2024.jsonl",
        help="Output JSONL path",
    )
    args = ap.parse_args()

    field_map: Dict[str, str] | None = None
    if args.field_map:
        field_map = json.loads(args.field_map)

    try:
        raw = load_dataset_generic(args.dataset, split=args.split)
    except Exception:
        # Fallback to HF Hub parquet loader
        raw = load_dataset_via_hub_parquet(args.dataset)
    examples = normalize_aime_examples(raw, field_map=field_map)
    rows = to_jsonl_rows(examples)
    save_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
