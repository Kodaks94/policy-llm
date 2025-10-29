from policy_rag.validation.grounding_check import judge_answer, enforce_grounding

def test_judge_and_enforce():
    retrieval_ctx = {
        "query": "data breach",
        "results": [{
            "chunk_id": "1",
            "text": "Any data breach must be reported immediately to the DPO.",
            "metadata": {}
        }]
    }
    answer_obj = {
        "answer": "Report to DPO immediately.",
        "citations": [],
        "is_fully_grounded": True
    }

    report = judge_answer(answer_obj, retrieval_ctx)
    safe_ans = enforce_grounding(answer_obj, report)
    assert "hallucination_flag" in safe_ans
