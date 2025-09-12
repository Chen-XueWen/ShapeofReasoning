import re


_STEP_PATTERNS = [
    re.compile(r"^\s*(?:step\s*\d+\s*[:\.-]|\(\d+\)|\d+\.|\d+\))\s*", re.I),
]


def _looks_like_step_start(s: str) -> bool:
    return any(pat.search(s) for pat in _STEP_PATTERNS)


def segment_steps(text: str, min_len: int = 2) -> list[str]:
    """
    Heuristic segmentation of reasoning into steps.

    Priority:
    - Respect explicit step markers ("Step 1:", "1.", "(1)") when present.
    - Otherwise, split by sentences as a fallback.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def segment_solution_steps(text: str) -> list[str]:
    """
    Segment solution text into steps
    """
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return [s for s in sentences]
