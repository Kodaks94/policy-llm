import os
from datetime import datetime

# ---- MODEL CHOICES ----
# These are Hugging Face models you can swap for bigger ones.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_GENERATION_MODEL = os.getenv("LLM_GENERATION_MODEL", "distilgpt2")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "distilgpt2")

# ---- VECTOR DB ----
VECTOR_STORE_PATH = "./vector_store.npz"
METADATA_STORE_PATH = "./chunk_metadata.jsonl"

# ---- RAG SETTINGS ----
TOP_K = 5
MIN_CHUNK_WORDS = 60
MAX_CHUNK_TOKENS = 800  # soft budget
CHUNK_OVERLAP_SENTENCES = 2

# ---- LOGGING ----
WANDB_PROJECT = "policy-rag"
ENV = os.getenv("APP_ENV", "dev")

def now_iso():
    return datetime.utcnow().isoformat() + "Z"
