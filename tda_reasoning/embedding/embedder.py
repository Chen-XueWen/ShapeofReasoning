from dataclasses import dataclass
from typing import Iterable, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: Optional[str] = None  # e.g., "cpu" or "cuda"


class SentenceTransformerEmbedder:
    """
    Thin wrapper around sentence-transformers with lazy import and helpful errors.
    """

    def __init__(self, cfg: EmbeddingConfig | None = None):
        self.cfg = cfg or EmbeddingConfig()
        self.model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        arr = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(arr)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T
