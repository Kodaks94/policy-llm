from policy_rag.chunking.chunker import build_chunks

def test_build_chunks(loaded_doc):
    chunks = build_chunks(loaded_doc)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c
        assert isinstance(c["text"], str)
