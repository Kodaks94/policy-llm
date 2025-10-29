from ..models.embed_client import get_embedding_matrix
from ..config import TOP_K

def retrieve_relevant_chunks(query: str, vs):
    q_emb = get_embedding_matrix([query])[0]  # (D,)
    raw_results = vs.search(q_emb, k=TOP_K)

    return {
        "query": query,
        "retriever_version": "cosine_v1",
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": r["score"],
                "metadata": r["metadata"],
            } for r in raw_results
        ]
    }
