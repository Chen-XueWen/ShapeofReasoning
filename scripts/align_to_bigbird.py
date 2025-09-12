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




def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_sentence_transformer():
    """
    Load BigBird model for computing similarities with large attention window.
    BigBird can handle up to 4096 tokens, perfect for longer text comparisons.
    """

    # Load BigBird model and tokenizer
    model_name = 'google/bigbird-roberta-base'
    configuration = BigBirdConfig(attention_type="original_full")
    tokenizer = BigBirdTokenizer.from_pretrained(model_name)
    model = BigBirdModel(configuration)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Loaded BigBird model: {model_name} (supports up to 4096 tokens)")
    return {'model': model, 'tokenizer': tokenizer}


def get_text_embedding(text, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    
    # Tokenize with truncation to handle very long texts
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                        max_length=4096, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use the [CLS] token embedding (first token) as text representation

        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    
    return embedding
        
   

def main() -> None:
    ap = argparse.ArgumentParser(description="Align model to bigbird")
    ap.add_argument("--aime", default="data/raw/aime2024.jsonl", help="AIME JSONL with gold text")
    ap.add_argument("--traces", default="data/raw/traces_gpt-oss_20b_1.jsonl", help="Model traces JSONL")
    ap.add_argument("--out", default="data/processed/gpt-oss_20b_1_alignments_bigbird.jsonl", help="Output JSONL")
    args = ap.parse_args()

    ensure_dir(args.out)

    bigbird = load_sentence_transformer()

    gold_by_id: Dict[str, List[str]] = {}
    for ex in read_jsonl(args.aime):
        # gold_steps = ex.get("gold_steps") ###todo: verify that this actually gets the solution instead of null gold_steps
        # if not gold_steps and ex.get("solution"):
        #     gold_steps = segment_steps(ex["solution"])  # best-effort
        gold_standard = ex.get("solution")
        gold_by_id[str(ex.get("id"))] = gold_standard

    out_f = open(args.out, "w", encoding="utf-8")
    n = 0
    for tr in tqdm(read_jsonl(args.traces), desc="Aligning traces", unit="trace"):
        pid = str(tr.get("id"))
        model_text = tr.get("trace", "")

        gold_standard = gold_by_id.get(pid)
        model_embedding = get_text_embedding(model_text, bigbird)
        golden_embedding = get_text_embedding(gold_standard, bigbird)

        similarity = cosine_similarity(model_embedding, golden_embedding)[0][0]
        
        similarity = max(0.0, similarity)
        row = {
            "id": pid,
            "score": float(similarity),
        }
        out_f.write(json.dumps(row) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote {n} alignments to {args.out}")


if __name__ == "__main__":
    main()
