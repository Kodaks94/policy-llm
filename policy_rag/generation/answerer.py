import json
from ..models.llm_client import call_llm

SYSTEM_PROMPT = (
    "You are a compliance assistant.\n"
    "ONLY use the provided policy context to answer.\n"
    "If not explicitly stated, reply exactly: \"Not covered by current policy.\"\n"
    "Always include citations from the provided context.\n"
    "Return valid JSON."
)

def build_context_block(results: list[dict]) -> str:
    blocks = []
    for r in results:
        clause = r["metadata"].get("approx_clause_ref")
        source = r["metadata"].get("source_id")
        block = (
            f"[chunk_id: {r['chunk_id']} | clause: {clause} | source: {source}]\n"
            f"{r['text']}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)

def answer_question(query: str, retrieval_ctx: dict) -> dict:
    context_text = build_context_block(retrieval_ctx["results"])

    user_prompt = f"""
Question:
{query}

Policy Context:
{context_text}

Respond in JSON with keys:
- answer: string
- citations: array of objects with keys clause_ref, source_id, chunk_id
- is_fully_grounded: boolean
"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt)

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "answer": "Not covered by current policy.",
            "citations": [],
            "is_fully_grounded": False
        }

    parsed.setdefault("answer", "Not covered by current policy.")
    parsed.setdefault("citations", [])
    parsed.setdefault("is_fully_grounded", False)

    # fallback citations using retrieved chunks
    if not parsed["citations"]:
        fallback = []
        for r in retrieval_ctx["results"]:
            fallback.append({
                "clause_ref": r["metadata"].get("approx_clause_ref"),
                "source_id": r["metadata"].get("source_id"),
                "chunk_id": r["chunk_id"]
            })
        parsed["citations"] = fallback

    return parsed
