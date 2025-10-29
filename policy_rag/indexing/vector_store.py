import os, json
import numpy as np
from ..config import VECTOR_STORE_PATH, METADATA_STORE_PATH

class VectorStore:
    """
    Simple cosine-similarity store backed by numpy arrays.
    Keeps:
      - self.embeddings: N x D float32
      - self.metadata:  list[dict] aligned row-wise
    """

    def __init__(self, embeddings=None, metadata=None):
        if embeddings is None:
            self.embeddings = np.zeros((0,0), dtype="float32")
        else:
            self.embeddings = embeddings.astype("float32")
        self.metadata = metadata if metadata is not None else []

    def add(self, new_embs: np.ndarray, metadatas: list[dict]):
        new_embs = new_embs.astype("float32")
        if self.embeddings.size == 0:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])
        self.metadata.extend(metadatas)

    def search(self, query_emb: np.ndarray, k: int):
        if self.embeddings.size == 0:
            return []
        A = self.embeddings      # (N, D)
        q = query_emb.astype("float32")  # (D,)

        qn = np.linalg.norm(q) + 1e-12
        An = np.linalg.norm(A, axis=1) + 1e-12
        sims = (A @ q) / (An * qn)  # cosine similarity

        idxs = np.argsort(-sims)[:k]
        results = []
        for idx in idxs:
            md = dict(self.metadata[idx])
            md["score"] = float(sims[idx])
            results.append(md)
        return results

    def save(self):
        np.savez(VECTOR_STORE_PATH, embeddings=self.embeddings)
        with open(METADATA_STORE_PATH, "w") as f:
            for md in self.metadata:
                f.write(json.dumps(md) + "\n")

    @classmethod
    def load(cls):
        if os.path.exists(VECTOR_STORE_PATH):
            data = np.load(VECTOR_STORE_PATH, allow_pickle=True)
            embeddings = data["embeddings"]
        else:
            embeddings = np.zeros((0,0), dtype="float32")

        metadata = []
        if os.path.exists(METADATA_STORE_PATH):
            with open(METADATA_STORE_PATH) as f:
                for line in f:
                    metadata.append(json.loads(line))
        return cls(embeddings=embeddings, metadata=metadata)

    @property
    def dim(self):
        if self.embeddings.size == 0:
            return 0
        return self.embeddings.shape[1]
