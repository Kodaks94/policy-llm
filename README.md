# policy-llm

#  Policy-RAG: Automated Policy QA & Validation Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that ingests policy PDFs, validates and cleans them, chunks and embeds content, retrieves context, and uses a **Hugging Face LLM** to answer questions — all while detecting hallucinations and logging metrics to **Weights & Biases (wandb)**.

---

##  Features

 **End-to-End Pipeline**
- PDF ingestion and text extraction (via PyMuPDF)  
- Automatic format and PII validation  
- Clause-based chunking and semantic embedding  
- Vector search (cosine-based retrieval)  
- Hugging Face LLM for grounded policy Q&A  
- Hallucination detection and grounding enforcement  
- Retrieval & generation metrics  
- Full observability with Weights & Biases  

 **LLM + Embeddings (Hugging Face)**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generator: `distilgpt2` (easily replaceable)
- Judge model: same or custom

 **FastAPI API Layer**
- `/ingest` → add and validate new policy PDFs  
- `/ask` → query your indexed policy collection  

 **Evaluation**
- Computes precision@k, recall@k, MRR, faithfulness, context precision, and answer relevance.  
- Drift-ready — just re-run eval periodically to detect performance degradation.

---

## Requirements

- Python ≥ 3.9  
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

**Key packages:**
- `fastapi`, `uvicorn` – web API  
- `PyMuPDF` – PDF extraction  
- `sentence-transformers`, `transformers`, `torch` – LLM + embeddings  
- `numpy` – vector math  
- `wandb` – logging and observability  

---

##  Running the API

Start the API server:
```bash
uvicorn policy_rag.app:app --reload
```

###  Ingest a Policy PDF

```bash
curl -X POST http://127.0.0.1:8000/ingest   -H "Content-Type: application/json"   -d '{"pdf_path":"./Employee_Conduct_Policy.pdf"}'
```

###  Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/ask   -H "Content-Type: application/json"   -d '{"question":"What should I do if a data breach occurs?"}'
```

---

### Sample outputs

 [Sample Output PDF](./Policy_RAG_Sample_Output.pdf)


## 📜 License
MIT License © 2025
