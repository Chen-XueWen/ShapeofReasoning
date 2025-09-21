#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, List

import re

import requests
from tqdm import tqdm


def load_aime_dataset(path: str | Path) -> List[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows: List[dict[str, Any]] = []
    for entry in data:
        combined_solution = combine_solutions(entry.get("solutions") or [])
        final_answer = entry.get("final_solution")
        final_answer = str(final_answer).strip() if final_answer is not None else None
        rows.append(
            {
                "id": entry.get("year-set-id"),
                "statement": str(entry.get("problem", "")),
                "final_answer": final_answer,
                "solution": combined_solution,
            }
        )
    return rows


def combine_solutions(solutions: List[dict[str, Any]]) -> str:
    parts: List[str] = []
    for solution in solutions:
        title = (solution.get("title") or "").strip()
        content = (solution.get("content") or "").strip()
        if title and content:
            parts.append(f"{title}:\n{content}")
        elif content:
            parts.append(content)
        elif title:
            parts.append(title)
    return "\n\n".join(parts).strip()


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
    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json() or {}
    # Some models may place content in 'thinking' (or 'reasoning') and leave 'response' empty
    response = data.get("response") or ""
    thinking = data.get("thinking") or data.get("reasoning") or ""
    text = (thinking + "\n" if thinking else "") + response
    return text.strip()



SYS_PROMPT_WITH_ANSWER = (
    "You are an expert competition mathematician. You are given the correct final numeric answer. "
    "Produce a clear, rigorous step-by-step solution that leads to this answer. "
    "Do not change the answer. Ensure each step is justified and consistent. "
    "End with the final line formatted exactly as 'Final Answer: <number>' using the provided answer."
)

SYS_PROMPT_WITHOUT_ANSWER = (
    "You are an expert competition mathematician. Solve the problem carefully and present a clear, rigorous "
    "step-by-step solution. Ensure each step is justified and consistent. End with the final line formatted "
    "exactly as 'Final Answer: <number>'."
)


FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*([^\n\r]+)", re.IGNORECASE)


def extract_final_answer(text: str) -> str | None:
    matches = FINAL_ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def build_prompt(statement: str, final_answer: str | None, include_final_answer: bool) -> str:
    if include_final_answer:
        if final_answer is None or str(final_answer).strip() == "":
            raise ValueError("final_answer is required in the dataset when include_final_answer=True")
        ans = str(final_answer).strip()
        return (
            f"{SYS_PROMPT_WITH_ANSWER}\n\n"
            f"Problem:\n{statement}\n\n"
            f"Provided Final Answer: {ans}\n\n"
            f"Solution:"
        )
    return (
        f"{SYS_PROMPT_WITHOUT_ANSWER}\n\n"
        f"Problem:\n{statement}\n\n"
        f"Solution:"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reasoning traces per problem via Ollama")
    ap.add_argument("--aime", default="data/aime_json/aime2024.json", help="AIME JSON input")
    ap.add_argument("--out", default="data/aime_traces/gptoss20b/traces_aime2024.jsonl", help="Output JSONL")
    ap.add_argument("--model", default="gpt-oss:20b", help="Ollama model name/tag")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
    ap.add_argument("--max_tokens", type=int, default=4096, help="Max tokens to generate")
    #ap.add_argument("--think-level", 
    #                default="medium", 
    #                help="think_level for qwen3/deepseek-r1 models must be True or False, for gpt-oss models must be one of 'low', 'medium', 'high'")
    ap.add_argument(
        "--include-final-answer",
        action="store_true",
        default=False,
        help="Include the known final answer in the generated prompt",
    )
    args = ap.parse_args()

    ensure_dir(args.out)
    out_f = open(args.out, "w", encoding="utf-8")

    options = {
        "temperature": 0.0,
        "seed": args.seed,
        "num_predict": args.max_tokens,
    }

    # Read all rows to enable a progress bar with total (dataset is small)
    rows: list[dict[str, Any]] = load_aime_dataset(args.aime)
    n = 0
    for ex in tqdm(rows, desc="Generating traces", unit="problem"):
        statement = ex.get("statement", "")
        pid = ex.get("id")
        final_answer = ex.get("final_answer")
        prompt = build_prompt(
            statement,
            final_answer=final_answer,
            include_final_answer=args.include_final_answer,
        )
        try:
            resp = ollama_generate(args.model, prompt, options=options, host=args.host)
        except Exception as e:
            resp = f"<ERROR: {e}>"
        pred_answer = extract_final_answer(resp)
        row = {
            "id": pid,
            "statement": statement,
            "model": args.model,
            "seed": args.seed,
            "temperature": 0.0,
            "trace": resp,
            "timestamp": int(time.time()),
            "given_answer": None,
            "pred_answer": None,
            "reference_solution": None,
        }
        row["pred_answer"] = pred_answer
        if final_answer is not None:
            row["given_answer"] = str(final_answer)
        if ex.get("solution"):
            row["reference_solution"] = ex["solution"]
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    out_f.close()
    print(f"Wrote {n} traces to {args.out}")


if __name__ == "__main__":
    main()
