#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm
import requests


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ollama_generate(
    model: str,
    prompt: str,
    options: dict[str, Any] | None = None,
    host: str = "http://localhost:11434",
) -> str:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json() or {}
    # Some models may place content in 'thinking' (or 'reasoning') and leave 'response' empty
    response = data.get("response") or ""
    thinking = data.get("thinking") or data.get("reasoning") or ""
    text = (thinking + "\n" if thinking else "") + response
    return text.strip()



SYS_PROMPT_GIVEN_ANSWER = (
    "You are an expert competition mathematician. You are given the correct final numeric answer. "
    "Produce a clear, rigorous step-by-step solution that leads to this answer. "
    "Do not change the answer. Ensure each step is justified and consistent. "
    "End with the final line formatted exactly as 'Final Answer: <number>' using the provided answer."
)


def build_prompt(statement: str, final_answer: str | None) -> str:
    if final_answer is None or str(final_answer).strip() == "":
        raise ValueError("final_answer is required in the dataset for prompt construction")
    ans = str(final_answer).strip()
    return (
        f"{SYS_PROMPT_GIVEN_ANSWER}\n\n"
        f"Problem:\n{statement}\n\n"
        f"Provided Final Answer: {ans}\n\n"
        f"Solution:"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 1 CoT trace per problem via Ollama (justify given final answer)")
    ap.add_argument("--aime", default="data/raw/aime2024.jsonl", help="AIME JSONL input")
    ap.add_argument("--out", default="data/raw/traces_aime2024.jsonl", help="Output JSONL")
    ap.add_argument("--model", default="gpt-oss:20b", help="Ollama model name/tag")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
    ap.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    args = ap.parse_args()

    ensure_dir(args.out)
    out_f = open(args.out, "w", encoding="utf-8")

    # Options: temperature 0 for reproducibility
    options = {
        "temperature": 0.0,
        "seed": args.seed,
        "num_predict": args.max_tokens,
    }

    # Read all rows to enable a progress bar with total (dataset is small)
    rows: list[dict[str, Any]] = list(read_jsonl(args.aime))
    n = 0
    for ex in tqdm(rows, desc="Generating traces", unit="problem"):
        statement = ex.get("statement", "")
        pid = ex.get("id")
        final_answer = ex.get("final_answer")
        prompt = build_prompt(statement, final_answer=final_answer)
        try:
            resp = ollama_generate(args.model, prompt, options=options, host=args.host)
        except Exception as e:
            resp = f"<ERROR: {e}>"
        row = {
            "id": pid,
            "statement": statement,
            "model": args.model,
            "seed": args.seed,
            "temperature": 0.0,
            "trace": resp,
            "timestamp": int(time.time()),
        }
        if final_answer is not None:
            row["given_answer"] = str(final_answer)
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote {n} traces to {args.out}")


if __name__ == "__main__":
    main()
