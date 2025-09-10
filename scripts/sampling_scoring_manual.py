#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path when running from scripts/
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from tqdm import tqdm

from tda_reasoning.embedding.segment import segment_steps
from tda_reasoning.embedding.embedder import EmbeddingConfig, SentenceTransformerEmbedder
from tda_reasoning.eval.align import align_steps
from transformers import BigBirdTokenizer, BigBirdModel, BigBirdConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd




def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
        

def main() -> None:
    ap = argparse.ArgumentParser(description="Scoring 10 traces manually")
    ap.add_argument("--aime", default="data/raw/aime2024.jsonl", help="AIME JSONL with gold text")
    ap.add_argument("--traces", default="data/raw/traces_gpt-oss_120b.jsonl", help="Model traces JSONL")
    ap.add_argument("--out", default="data/processed/alignments_sampling_scores.jsonl", help="Output JSONL")
    args = ap.parse_args()

    ensure_dir(args.out)

    gold_by_id: Dict[str, List[str]] = {}
    for ex in read_jsonl(args.aime):
        answer = ex.get("final_answer")
        gold_by_id[str(ex.get("id"))] = answer

    out_f = open(args.out, "w", encoding="utf-8")
    n = 0

    scores = {}
    all_df = pd.DataFrame()
    for i in range(1, 11): 
        trace_path = args.traces.replace(".jsonl", f"_{i}.jsonl")
        cur_series = {}
        for tr in tqdm(read_jsonl(args.traces), desc="sampling", unit="trace"):
            pid = str(tr.get("id"))
            model_text = tr.get("trace", "")
            answer = gold_by_id.get(pid)
            print(model_text[-500:])
            print(pid)
            print("GOLD ANSWER: " + answer)
            print("is the answer right? (y/n)")

            trace_score = 1 if input().strip().lower() == "y" else 0 #use pandas to add this to a dataframe for each qn so we don't lose the info
            # trace_score = 1
            cur_series[pid] = trace_score
            #update dictionary
            cur_scores = scores.get(pid, 0)
            scores[pid] = cur_scores + trace_score
        all_df[i] = cur_series
    
    all_df.to_csv("data/processed/sampling_scores.csv")
    for pid, score in scores.items():
        score = score / 10.0
        row = {
            "id": pid,
            "score": float(score),
        }
        out_f.write(json.dumps(row) + "\n")
    out_f.close()
    print(f"Wrote {n} alignments to {args.out}")


if __name__ == "__main__":
    main()
