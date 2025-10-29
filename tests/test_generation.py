from policy_rag.generation.answerer import answer_question

def test_answer_generation(indexed_store):
    retrieval_ctx = {
        "query": "data breach",
        "results": [{
            "chunk_id": "sample::1",
            "text": "Any data breach must be reported to the DPO within 72 hours.",
            "metadata": {"source_id": "sample.pdf", "approx_clause_ref": "5.1"}
        }]
    }
    ans = answer_question("What should I do if a data breach occurs?", retrieval_ctx)
    assert "answer" in ans
    assert "citations" in ans
