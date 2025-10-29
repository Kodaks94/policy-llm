import re
from ..config import MIN_CHUNK_WORDS, CHUNK_OVERLAP_SENTENCES

# We split on numbered clauses like "4.2", "3.1.5", etc.
CLAUSE_SPLIT_RE = r"(?=(?:^|\n)\s*(\d+(?:\.\d+)*\s+))"

def split_into_clauses(full_text: str):
    parts = re.split(CLAUSE_SPLIT_RE, full_text)
    rebuilt = []
    i = 0
    while i < len(parts):
        block = parts[i].strip()
        if re.match(r"^\d+(?:\.\d+)*\s*$", block):
            clause_number = block
            clause_body = parts[i+1].strip() if i + 1 < len(parts) else ""
            rebuilt.append(f"{clause_number} {clause_body}")
            i += 2
        else:
            if block:
                rebuilt.append(block)
            i += 1
    return rebuilt

def sentence_tokenize(text: str):
    # crude sentence splitter
    return re.split(r'(?<=[.!?])\s+', text.strip())

def merge_small_clauses(clauses):
    merged = []
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            merged.append(" ".join(buffer).strip())
            buffer = []

    for c in clauses:
        word_count = len(c.split())
        if word_count < MIN_CHUNK_WORDS:
            buffer.append(c)
        else:
            flush()
            merged.append(c)
    flush()
    return merged

def add_overlap(chunks):
    final = []
    for idx, chunk in enumerate(chunks):
        sents = sentence_tokenize(chunk)
        if idx > 0:
            prev_tail = sentence_tokenize(chunks[idx-1])[-CHUNK_OVERLAP_SENTENCES:]
            chunk_with_overlap = " ".join(prev_tail + sents)
        else:
            chunk_with_overlap = chunk
        final.append(chunk_with_overlap.strip())
    return final

def extract_clause_ref(text: str):
    m = re.match(r"^(\d+(?:\.\d+)*)", text.strip())
    return m.group(1) if m else None

def build_chunks(doc_batch):
    """
    doc_batch from load_pdf().
    Output: list[Chunk], where each chunk looks like:
    {
      "chunk_id": "Policy.pdf::7",
      "text": "...chunk text...",
      "metadata": {
         "source_id": "...pdf filename...",
         "ingestion_timestamp": "...",
         "hash": "...",
         "approx_clause_ref": "4.2"
      }
    }
    """
    full_text = "\n\n".join([p["text"] for p in doc_batch["pages"]])
    clauses = split_into_clauses(full_text)
    merged = merge_small_clauses(clauses)
    overlapped = add_overlap(merged)

    chunks = []
    for i, text in enumerate(overlapped):
        chunks.append({
            "chunk_id": f"{doc_batch['source_id']}::{i}",
            "text": text,
            "metadata": {
                "source_id": doc_batch["source_id"],
                "ingestion_timestamp": doc_batch["ingestion_timestamp"],
                "hash": doc_batch["hash"],
                "approx_clause_ref": extract_clause_ref(text)
            }
        })
    return chunks
