import argparse
import json
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser(description="plot and compare TDA features")
    ap.add_argument("--feature-files", nargs="+", required=True, help="TDA feature JSONL files to compare")
    ap.add_argument("--feature-names", nargs="+", required=True, help="Names for the features")

    args = ap.parse_args()

    if args.feature_names is None or len(args.feature_names) == 0:
        raise ValueError("At least one feature name must be provided.")
    if args.feature_files is None or len(args.feature_files) == 0:
        raise ValueError("At least one feature file must be provided.")

    features = {file: {
        name: [] for name in args.feature_names
    } for file in args.feature_files}

    for file in tqdm(args.feature_files):
        # Process each feature file
        for item in read_jsonl(file):
            for name in args.feature_names:
                if name in item:
                    features[file][name].append(item[name])
    
    for name in args.feature_names:
        for file in args.feature_files:
            print(f"Mean of {name} for {file}:", sum(features[file][name]) / len(features[file][name]) if features[file][name] else 0)
        plt.hist(
            [features[file][name] for file in args.feature_files],
            label=args.feature_files,
            alpha=0.5,
        )
        plt.title(f"Histogram of {name}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()