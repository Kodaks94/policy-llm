from ..models.embed_client import get_embedding_matrix
from .vector_store import VectorStore

def ingest_chunks_into_index(chunks: list[dict], vs: VectorStore | None = None) -> VectorStore:
    """
    chunks -> embed -> add to vector store -> save.
    """
    texts = [c["text"] for c in chunks]
    embs = get_embedding_matrix(texts)  # (N, D)

    metadatas = []
    for c in chunks:
        metadatas.append({
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "metadata": c["metadata"]
        })

    if vs is None:
        vs = VectorStore()

    vs.add(embs, metadatas)
    vs.save()
    return vs
