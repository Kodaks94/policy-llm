import numpy as np
from policy_rag.indexing.embed_and_store import ingest_chunks_into_index
from policy_rag.models.embed_client import get_embedding_matrix
from policy_rag.chunking.chunker import build_chunks
from policy_rag.indexing.vector_store import VectorStore

def test_embeddings_shape(loaded_doc):
    chunks = build_chunks(loaded_doc)
    texts = [c["text"] for c in chunks]
    embs = get_embedding_matrix(texts)
    assert embs.ndim == 2
    assert embs.shape[0] == len(texts)

def test_vector_store_add(loaded_doc):
    chunks = build_chunks(loaded_doc)
    vs = VectorStore()
    vs = ingest_chunks_into_index(chunks, vs)
    assert vs.dim > 0
    assert len(vs.metadata) == len(chunks)
