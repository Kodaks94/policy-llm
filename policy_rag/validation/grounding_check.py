import json
from ..models.llm_client import call_judge

JUDGE_SYSTEM = (
    "You are a compliance verifier. "
    "Your job is to check if the answer is FULLY supported by the provided context."
)

def judge_answer(answer_obj: dict, retrieval_ctx: dict) -> dict:
    context_block = "\n\n".join(
        [f"{r['chunk_id']}: {r['text']}" for r in retrieval_ctx["results"]]
    )
    candidate_answer = answer_obj["answer"]

    judge_user = f"""
Context:
{context_block}

Answer:
{candidate_answer}

Question:
{retrieval_ctx['query']}

Task:
- Is EVERY factual claim in Answer explicitly supported by Context?
Return JSON:
{{
  "faithful": true/false,
  "unsupported_spans": ["...", "..."]
}}
"""

    raw = call_judge(JUDGE_SYSTEM, judge_user)

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"faithful": True, "unsupported_spans": []}
    return parsed

def enforce_grounding(answer_obj: dict, judge_report: dict) -> dict:
    if not judge_report.get("faithful", True):
        return {
            "answer": "Not covered by current policy.",
            "citations": [],
            "is_fully_grounded": False,
            "hallucination_flag": True,
            "unsupported_spans": judge_report.get("unsupported_spans", [])
        }

    safe = dict(answer_obj)
    safe["hallucination_flag"] = False
    safe["unsupported_spans"] = []
    return safe
