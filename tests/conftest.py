import pytest
from policy_rag.ingestion.pdf_loader import load_pdf
from policy_rag.chunking.chunker import build_chunks
from policy_rag.indexing.embed_and_store import ingest_chunks_into_index
from policy_rag.indexing.vector_store import VectorStore
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

@pytest.fixture(scope="session")
def sample_pdf_path():
    return "data/Data_Protection_Privacy_Policy.pdf"

@pytest.fixture(scope="session")
def loaded_doc(sample_pdf_path):
    return load_pdf(sample_pdf_path)

@pytest.fixture(scope="session")
def indexed_store(loaded_doc):
    chunks = build_chunks(loaded_doc)
    vs = ingest_chunks_into_index(chunks)
    return vs
