import time
import wandb
from ..config import WANDB_PROJECT, ENV

def init_run(run_type: str, meta: dict):
    wandb.init(
        project=WANDB_PROJECT,
        job_type=run_type,
        config={
            "env": ENV,
            **meta
        }
    )

def log_ingestion(doc_batch, format_report, pii_report):
    wandb.log({
        "ts": time.time(),
        "stage": "ingestion",
        "source_id": doc_batch["source_id"],
        "pii_severity": pii_report["severity"],
        "pii_action": pii_report["action"],
        "policy_valid": format_report["is_valid"],
        "missing_sections_count": len(format_report["missing_sections"])
    })

def log_query(query, retrieval_ctx, final_answer, judge_report):
    wandb.log({
        "ts": time.time(),
        "stage": "qa",
        "query": query,
        "retriever_version": retrieval_ctx["retriever_version"],
        "top_scores": [r["score"] for r in retrieval_ctx["results"]],
        "is_fully_grounded": final_answer["is_fully_grounded"],
        "hallucination_flag": final_answer.get("hallucination_flag", False),
        "unsupported_spans": final_answer.get("unsupported_spans", [])
    })

def log_eval(batch_metrics):
    wandb.log({
        "ts": time.time(),
        "stage": "eval",
        **batch_metrics
    })
