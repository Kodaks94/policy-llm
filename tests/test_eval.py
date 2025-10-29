from policy_rag.eval.evaluator import aggregate_eval

def test_aggregate_eval():
    fake_records = [{
        "relevant_chunk_ids": ["1","2"],
        "retrieved_chunk_ids": ["1","3"],
        "faithfulness": 0.9,
        "answer_relevance": 0.85,
        "context_precision": 0.88
    }]
    metrics = aggregate_eval(fake_records)
    assert 0 <= metrics["precision_at_5"] <= 1
    assert 0 <= metrics["faithfulness"] <= 1
