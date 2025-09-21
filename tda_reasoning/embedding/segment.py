import re


def _strip_math_inline_markers(text: str) -> str:
    """Remove lightweight LaTeX inline math wrappers."""
    return text.replace("\\(", "").replace("\\)", "").replace("$", "").replace("\\[","").replace("\\]","").replace("\\","").replace("<think>","")


def segment_steps(text: str, min_len: int = 2) -> list[str]:
    if not text:
        return []
    cleaned = _strip_math_inline_markers(text)
    segments: list[str] = []
    for ln in cleaned.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        pieces = re.split(r"(?<=\.)\s+", ln)
        segments.extend(piece.strip() for piece in pieces if piece.strip())
    return segments

def segment_solution_steps(text: str) -> list[str]:
    """
    Segment solution text into steps
    """
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return [s for s in sentences]
