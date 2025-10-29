import fitz  # PyMuPDF
import hashlib
from ..config import now_iso

def compute_file_hash(bytes_data: bytes):
    h = hashlib.sha256()
    h.update(bytes_data)
    return h.hexdigest()

def load_pdf(pdf_path: str):
    """
    Extracts raw text per page so we can trace answers back to page numbers.
    Returns a DocumentBatch dict.
    """
    with open(pdf_path, "rb") as f:
        raw = f.read()

    doc = fitz.open(stream=raw, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,
            "text": text.strip()
        })

    return {
        "source_id": pdf_path.split("/")[-1],
        "pages": pages,
        "ingestion_timestamp": now_iso(),
        "hash": compute_file_hash(raw)
    }
