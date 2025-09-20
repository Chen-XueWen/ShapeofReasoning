#!/usr/bin/env python3
"""Extract AIME problems, solutions, and final answers into JSON files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

PROBLEM_FILE_RE = re.compile(
    r"(?P<year>20\d{2})_AIME_(?P<set>[IV]+)_Problem_(?P<number>\d{1,2})\.html$"
)
ANSWER_KEY_RE = re.compile(r"(?P<year>20\d{2})_AIME_(?P<set>[IV]+)_Answer_Key\.html$")
SET_ORDER = {"I": 1, "II": 2, "III": 3}


@dataclass
class ProblemData:
    year: str
    set_id: str
    number: int
    problem: str
    solutions: List[Dict[str, str]]
    final_answer: Optional[str]

    @property
    def year_set_number(self) -> str:
        return f"{self.year}-{self.set_id}-{self.number}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "year-set-id": self.year_set_number,
            "problem": self.problem,
            "solutions": self.solutions,
            "final_solution": self.final_answer,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root (defaults to script directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where aime{year}.json files are written (defaults to root)",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        help="Optional list of years to process (e.g. 2020 2021)",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = value.replace("\u200b", "")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+(\n)", r"\1", value)
    value = re.sub(r"(\n)[ \t]+", r"\1", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def preprocess_dom(soup: BeautifulSoup, root: Tag) -> None:
    for img in root.find_all("img"):
        alt = img.get("alt")
        if alt:
            img.replace_with(soup.new_string(alt))
        else:
            img.decompose()
    for br in root.find_all("br"):
        br.replace_with(soup.new_string("\n"))
    for edit in root.select("span.mw-editsection"):
        edit.decompose()


def collect_section_text(span: Tag) -> str:
    heading = span.find_parent(["h2", "h3"])
    if heading is None:
        return ""

    chunks: List[str] = []
    for sibling in heading.find_next_siblings():
        if isinstance(sibling, Comment):
            continue
        if isinstance(sibling, NavigableString):
            text = sibling.strip()
            if text:
                chunks.append(text)
            continue
        if sibling.name in {"h2", "h3"}:
            break
        if sibling.name in {"script", "style"}:
            continue
        text = sibling.get_text(" ", strip=True)
        if text:
            chunks.append(text)
    combined = "\n\n".join(chunks)
    return normalize_text(combined)


def extract_problem_and_solutions(path: Path) -> Tuple[str, List[Dict[str, str]]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        return "", []
    preprocess_dom(soup, content)

    problem_text = ""
    problem_span = content.find("span", id="Problem")
    if problem_span is not None:
        problem_text = collect_section_text(problem_span)

    solutions: List[Dict[str, str]] = []
    for span in content.find_all("span", class_="mw-headline"):
        title = span.get_text(strip=True)
        if not title.lower().startswith("solution"):
            continue
        text = collect_section_text(span)
        if text:
            solutions.append({"title": title, "content": text})
    return problem_text, solutions


def extract_answer_key(path: Path) -> List[str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        return []
    preprocess_dom(soup, content)
    answers: List[str] = []
    ordered_list = content.find("ol")
    if ordered_list is None:
        return []
    for item in ordered_list.find_all("li", recursive=False):
        text = item.get_text(" ", strip=True)
        if text:
            answers.append(normalize_text(text))
    return answers


def load_final_answers(answer_dir: Path) -> Dict[Tuple[str, str], List[str]]:
    mapping: Dict[Tuple[str, str], List[str]] = {}
    if not answer_dir.exists():
        return mapping
    for path in sorted(answer_dir.glob("*_Answer_Key.html")):
        match = ANSWER_KEY_RE.match(path.name)
        if not match:
            continue
        year = match.group("year")
        set_id = match.group("set")
        answers = extract_answer_key(path)
        if answers:
            mapping[(year, set_id)] = answers
    return mapping


def iter_problem_files(directory: Path) -> Iterable[Tuple[str, str, int, Path]]:
    for path in directory.glob("*.html"):
        match = PROBLEM_FILE_RE.match(path.name)
        if not match:
            continue
        year = match.group("year")
        set_id = match.group("set")
        number = int(match.group("number"))
        yield year, set_id, number, path


def build_year_entries(year: str, problem_dir: Path, answer_map: Dict[Tuple[str, str], List[str]]) -> List[ProblemData]:
    problem_records: List[ProblemData] = []
    collected = list(iter_problem_files(problem_dir))
    collected.sort(key=lambda x: (SET_ORDER.get(x[1], 99), x[2]))

    for _, set_id, number, path in collected:
        problem_text, solutions = extract_problem_and_solutions(path)
        answers = answer_map.get((year, set_id))
        final_answer: Optional[str] = None
        if answers and 0 < number <= len(answers):
            final_answer = answers[number - 1]
        record = ProblemData(
            year=year,
            set_id=set_id,
            number=number,
            problem=problem_text,
            solutions=solutions,
            final_answer=final_answer,
        )
        problem_records.append(record)
    return problem_records


def write_year_json(records: List[ProblemData], output_path: Path) -> None:
    data = [record.to_dict() for record in records]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    html_root = root / "aime_html"
    answer_dir = html_root / "aime_answer_keys"
    output_dir = args.output_dir or root

    if not html_root.exists():
        raise SystemExit(f"Cannot find aime_html directory under {root}")

    answer_map = load_final_answers(answer_dir)

    problem_dirs = [
        d for d in html_root.iterdir() if d.is_dir() and d.name.endswith("_aime_problems")
    ]
    if args.years:
        requested = set(args.years)
        problem_dirs = [
            d for d in problem_dirs if d.name.split("_", 1)[0] in requested
        ]

    problem_dirs.sort()
    for problem_dir in problem_dirs:
        year = problem_dir.name.split("_", 1)[0]
        records = build_year_entries(year, problem_dir, answer_map)
        output_path = output_dir / f"aime{year}.json"
        write_year_json(records, output_path)
        print(f"Wrote {output_path} ({len(records)} entries)")


if __name__ == "__main__":
    main()
