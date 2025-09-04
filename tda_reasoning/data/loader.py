import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class AIMEExample:
    id: str
    year: Optional[int]
    index: Optional[int]
    statement: str
    gold_steps: Optional[List[str]] = None
    final_answer: Optional[str] = None
    solution_text: Optional[str] = None


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: str | os.PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_dataset_generic(dataset_name: str, split: str = "train", **kwargs) -> List[Dict[str, Any]]:
    """
    Load a dataset via Hugging Face Datasets by name and split.

    Returns a list of dicts. Caller should map fields with `field_map`.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' library is required. Install with `pip install datasets`."
        ) from e

    ds = load_dataset(dataset_name, split=split, **kwargs)  # type: ignore
    return [dict(x) for x in ds]


def load_dataset_via_hub_parquet(dataset_name: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Lightweight loader using huggingface_hub to download a parquet file and read via polars.
    If path is None, attempts to infer from dataset card (cardData.data_files[0].path).
    """
    try:
        import requests  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        import polars as pl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Install requirements: `pip install huggingface_hub polars requests`"
        ) from e

    if path is None:
        api_url = f"https://huggingface.co/api/datasets/{dataset_name}"
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        info = r.json()
        path = None
        files = info.get("cardData", {}).get("data_files", [])
        if files:
            path = files[0].get("path")
        if not path:
            # Fallback: look for a parquet sibling
            for sib in info.get("siblings", []):
                fn = sib.get("rfilename", "")
                if fn.endswith(".parquet"):
                    path = fn
                    break
        if not path:
            raise RuntimeError("Could not infer data file path from dataset card")

    local = hf_hub_download(repo_id=dataset_name, filename=path, repo_type="dataset")
    df = pl.read_parquet(local)
    return df.to_dicts()


def normalize_aime_examples(
    raw_items: List[Dict[str, Any]],
    field_map: Optional[Dict[str, str]] = None,
) -> List[AIMEExample]:
    """
    Map raw Hugging Face fields into a normalized AIMEExample schema.

    field_map may specify keys: statement, solution, answer, year, index, steps.
    If steps are not given in the dataset, caller can populate later via parsing.
    """
    field_map = field_map or {}
    s_key = field_map.get("statement", "problem")
    sol_key = field_map.get("solution", "solution")
    ans_key = field_map.get("answer", "answer")
    year_key = field_map.get("year", "year")
    idx_key = field_map.get("index", "index")
    steps_key = field_map.get("steps", "steps")

    out: List[AIMEExample] = []
    for i, it in enumerate(raw_items):
        # Accept case-variant keys (e.g., "Problem")
        def get_any(d: Dict[str, Any], key: str) -> Any:
            if key in d:
                return d[key]
            # try case-insensitive match
            for k in d.keys():
                if isinstance(k, str) and k.lower() == key.lower():
                    return d[k]
            return None

        statement = get_any(it, s_key) or get_any(it, "question") or get_any(it, "prompt") or ""
        solution = get_any(it, sol_key)
        answer = get_any(it, ans_key)
        steps = it.get(steps_key)
        year = it.get(year_key)
        index = it.get(idx_key)

        ex_id = get_any(it, "id") or get_any(it, "ID") or f"aime-{i}"
        example = AIMEExample(
            id=str(ex_id),
            year=int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
            index=int(index) if isinstance(index, (int, str)) and str(index).isdigit() else None,
            statement=str(statement).strip(),
            gold_steps=[str(s).strip() for s in steps] if isinstance(steps, list) else None,
            final_answer=str(answer).strip() if answer is not None else None,
            solution_text=str(solution).strip() if solution is not None else None,
        )

        # If only a single free-form solution string exists, keep it for later segmentation
        if example.gold_steps is None and solution:
            # Keep raw solution for later parsing; not splitting here
            example.gold_steps = None  # to be populated by downstream segmentation
            # Temporarily stash solution in final_answer if answer absent
            if example.final_answer is None:
                example.final_answer = None

        out.append(example)
    return out


def to_jsonl_rows(examples: List[AIMEExample]) -> List[Dict[str, Any]]:
    rows = []
    for e in examples:
        rows.append(
            {
                "id": e.id,
                "year": e.year,
                "index": e.index,
                "statement": e.statement,
                "gold_steps": e.gold_steps,
                "final_answer": e.final_answer,
                "solution": e.solution_text,
            }
        )
    return rows
