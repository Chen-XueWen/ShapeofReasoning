from __future__ import annotations

import re
from typing import List


_STEP_PATTERNS = [
    re.compile(r"^\s*(?:step\s*\d+\s*[:\.-]|\(\d+\)|\d+\.|\d+\))\s*", re.I),
]


def _looks_like_step_start(s: str) -> bool:
    return any(pat.search(s) for pat in _STEP_PATTERNS)


def segment_steps(text: str, min_len: int = 2) -> List[str]:
    """
    Heuristic segmentation of reasoning into steps.

    Priority:
    - Respect explicit step markers ("Step 1:", "1.", "(1)") when present.
    - Otherwise, split by sentences as a fallback.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    has_markers = sum(_looks_like_step_start(ln) for ln in lines) >= 2
    if has_markers:
        segments: List[str] = []
        buf: List[str] = []
        for ln in lines:
            if _looks_like_step_start(ln) and buf:
                segments.append(" ".join(buf).strip())
                buf = [ln]
            else:
                buf.append(ln)
        if buf:
            segments.append(" ".join(buf).strip())
        return [s for s in segments if len(s.split()) >= min_len]

    # Fallback: sentence split on punctuation
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return [s for s in sentences if len(s.split()) >= min_len]

