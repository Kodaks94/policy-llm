from fastapi import FastAPI
from pydantic import BaseModel
from .ingestion.pdf_loader import load_pdf
from .validation.policy_format_check import check_policy_format
from .validation.pii_scan import scan_pii, redact_pii_text
from .chunking.chunker import build_chunks
from .indexing.embed_and_store import ingest_chunks_into_index
from .indexing.vector_store import VectorStore
from .retrieval.retriever import retrieve_relevant_chunks
from .generation.answerer import answer_question
from .validation.grounding_check import judge_answer, enforce_grounding
from .observability import wandb_logger

app = FastAPI(title="Policy RAG API")

# in-memory index cache (for demo)
vector_store_cache = {"vs": None}

class IngestRequest(BaseModel):
    pdf_path: str  # path on disk/server

class AskRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest_policy(req: IngestRequest):
    # 1. load PDF
    doc_batch = load_pdf(req.pdf_path)

    # 2. format + pii validation
    full_text = "\n\n".join([p["text"] for p in doc_batch["pages"]])
    fmt_report = check_policy_format(full_text)
    pii_report = scan_pii(doc_batch["pages"])

    if pii_report["action"] == "block_ingestion":
        wandb_logger.init_run("ingestion", {"source_id": doc_batch["source_id"]})
        wandb_logger.log_ingestion(doc_batch, fmt_report, pii_report)
        return {
            "status": "rejected",
            "reason": "high severity PII detected",
            "pii_report": pii_report,
            "format_report": fmt_report,
        }

    # optional redaction pass
    if pii_report["action"] == "redact":
        for p in doc_batch["pages"]:
            p["text"] = redact_pii_text(p["text"])

    # 3. chunk + embed + index
    chunks = build_chunks(doc_batch)
    vs = vector_store_cache["vs"]
    vs = ingest_chunks_into_index(chunks, vs)
    vector_store_cache["vs"] = vs

    # 4. log to wandb
    wandb_logger.init_run("ingestion", {"source_id": doc_batch["source_id"]})
    wandb_logger.log_ingestion(doc_batch, fmt_report, pii_report)

    return {
        "status": "ingested",
        "source_id": doc_batch["source_id"],
        "chunks_indexed": len(chunks),
        "pii_severity": pii_report["severity"],
        "format_valid": fmt_report["is_valid"],
        "missing_sections": fmt_report["missing_sections"],
    }

@app.post("/ask")
def ask_policy(req: AskRequest):
    vs = vector_store_cache["vs"]
    if vs is None or vs.dim == 0:
        return {"error": "no documents indexed yet"}

    # 1. retrieve
    retrieval_ctx = retrieve_relevant_chunks(req.question, vs)

    # 2. generate answer using only retrieved chunks
    answer_obj = answer_question(req.question, retrieval_ctx)

    # 3. judge hallucination
    judge_report = judge_answer(answer_obj, retrieval_ctx)
    final_answer = enforce_grounding(answer_obj, judge_report)

    # 4. log
    wandb_logger.init_run("qa_request", {"generator_model": "LLM_GENERATION_MODEL"})
    wandb_logger.log_query(req.question, retrieval_ctx, final_answer, judge_report)

    return final_answer
