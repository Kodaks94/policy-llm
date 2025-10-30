#Fixed-Size Token (or Word) Chunking
# Pros: Simple, uniform length
#️ Cons: Can split sentences/clauses mid-way, losing semantic boundaries
def fixed_size_chunk(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk.strip())
    return chunks


import re
#Sentence-Based Semantic Chunking
# Pros: Keeps semantic structure intact
# Cons: Uneven chunk sizes
def sentence_chunk(text, max_words=150):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], []
    word_count = 0

    for sent in sentences:
        w = len(sent.split())
        if word_count + w > max_words:
            chunks.append(" ".join(current))
            current, word_count = [sent], w
        else:
            current.append(sent)
            word_count += w
    if current:
        chunks.append(" ".join(current))
    return chunks


#Paragraph-Based Chunking (Structured Documents)
# Pros: Works well for structured policy/legal documents
# Cons: Relies on consistent formatting in the original document
def paragraph_chunk(text):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        # Merge small paras with previous one
        if chunks and len(p.split()) < 50:
            chunks[-1] += " " + p
        else:
            chunks.append(p)
    return chunks

#Heading-Aware Chunking (Section-based)
# Pros: Perfect for policy docs with numbered sections
# Cons: Needs clear numbering (e.g. “1.1”, “2.3”)
def heading_based_chunk(text):
    pattern = re.compile(r"(?P<header>(?:\d+\.)+\s[^\n]+)")
    matches = list(pattern.finditer(text))
    chunks = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        chunks.append(m.group("header") + "\n" + section)
    return chunks

#Similarity-Aware (Semantic Cohesion) Chunking
# Pros: Keeps logically related sentences together
# Cons: Slower (requires embeddings), GPU recommended
from sentence_transformers import SentenceTransformer, util

def semantic_chunk(text, similarity_threshold=0.7, max_sentences=5):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = re.split(r'(?<=[.!?]) +', text)
    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks, current = [sentences[0]], []
    for i in range(1, len(sentences)):
        sim = float(util.cos_sim(embeddings[i-1], embeddings[i]))
        if sim < similarity_threshold or len(current) >= max_sentences:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


# Recursive Character-Based (LangChain-Style)
# Pros: Great for mixed formatting / large text blocks
# Cons: Doesn’t align perfectly with semantics
def recursive_char_chunk(text, max_chars=1000, overlap=100):
    if len(text) <= max_chars:
        return [text]
    split_point = text.rfind('.', 0, max_chars)
    if split_point == -1:
        split_point = max_chars
    chunk = text[:split_point+1]
    next_text = text[max(0, split_point - overlap):]
    return [chunk.strip()] + recursive_char_chunk(next_text, max_chars, overlap)
