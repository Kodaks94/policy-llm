import numpy as np

def precision_at_k(relevant_ids, retrieved_ids, k):
    topk = retrieved_ids[:k]
    num_rel = sum(1 for x in topk if x in relevant_ids)
    return num_rel / max(k,1)

def recall_at_k(relevant_ids, retrieved_ids, k):
    topk = retrieved_ids[:k]
    num_rel = sum(1 for x in topk if x in relevant_ids)
    return num_rel / max(len(relevant_ids),1)

def mrr(relevant_ids, retrieved_ids):
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0

def aggregate_eval(eval_records):
    """
    eval_records: list of dicts with:
      relevant_chunk_ids
      retrieved_chunk_ids
      faithfulness
      answer_relevance
      context_precision
    """
    p5 = np.mean([precision_at_k(r["relevant_chunk_ids"], r["retrieved_chunk_ids"], 5) for r in eval_records])
    r5 = np.mean([recall_at_k(r["relevant_chunk_ids"], r["retrieved_chunk_ids"], 5) for r in eval_records])
    m  = np.mean([mrr(r["relevant_chunk_ids"], r["retrieved_chunk_ids"]) for r in eval_records])
    f  = np.mean([r["faithfulness"] for r in eval_records])
    ar = np.mean([r["answer_relevance"] for r in eval_records])
    cp = np.mean([r["context_precision"] for r in eval_records])

    return {
        "precision_at_5": float(p5),
        "recall_at_5": float(r5),
        "mrr": float(m),
        "faithfulness": float(f),
        "answer_relevance": float(ar),
        "context_precision": float(cp),
    }
