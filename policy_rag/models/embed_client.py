import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL

_embedder = None

def _load_model():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def get_embedding(text: str) -> list[float]:
    model = _load_model()
    emb = model.encode([text], convert_to_numpy=True)[0]
    return emb.tolist()

def get_embedding_matrix(texts: list[str]) -> np.ndarray:
    model = _load_model()
    mat = model.encode(texts, convert_to_numpy=True)
    return mat  # shape (N, D)
