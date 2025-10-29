from policy_rag.retrieval.retriever import retrieve_relevant_chunks

def test_retrieve(indexed_store):
    res = retrieve_relevant_chunks("data breach", indexed_store)
    assert "results" in res
    assert len(res["results"]) > 0
    assert "text" in res["results"][0]
