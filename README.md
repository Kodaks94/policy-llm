# Policy-RAG: Retrieval-Augmented Generation Framework for Policy Documents

## Overview

**Policy-RAG** is an end-to-end Retrieval-Augmented Generation (RAG) system designed to process and query corporate policy documents such as privacy, conduct, and HR policies.  
It performs text extraction, validation, chunking, embedding, retrieval, and LLM-based question answering, with integrated evaluation and observability features.

---

## Features

- Automatic PDF ingestion: Extracts and validates text content from uploaded policy PDFs.  
- PII detection and format validation: Flags sensitive data and checks for structural consistency.  
- Flexible text chunking: Multiple configurable strategies for optimal retrieval and context windows.  
- Embedding and retrieval: Uses transformer-based embeddings for similarity search.  
- LLM-powered response generation: Produces grounded answers using retrieved context.  
- LoRA fine-tuning support: Lightweight domain adaptation using PEFT.  
- Comprehensive evaluation: Faithfulness, grounding, relevance, and completeness via the OpenAI Evals framework.  
- Experiment tracking: Weights & Biases integration for model and evaluation logging.  
- Fully unit-tested: Verified functionality across ingestion, chunking, retrieval, and generation components.

---

## Pipeline Overview

1. Input: Policy PDF files (e.g., privacy, conduct, or HR policies).  
2. Validation: Performs format and PII checks.  
3. Chunking: Splits text into manageable, semantically coherent units.  
4. Embedding: Converts chunks into vector representations.  
5. Retrieval: Retrieves contextually relevant chunks for a given query.  
6. Generation: Uses an LLM (optionally LoRA-adapted) to produce a grounded response.  
7. Evaluation: Assesses output quality using the Evals library.  
8. Logging: Reports metrics to Weights & Biases.

---

## Alternative Chunking Strategies

Implemented under `policy_rag/chunking/alt_chunkers.py`

| Method | Strengths | Weaknesses | Best For |
|---------|------------|-------------|-----------|
| Fixed-size | Simple and consistent | Cuts sentences mid-way | Raw text |
| Sentence-based | Maintains sentence meaning | Uneven chunk sizes | General prose |
| Paragraph-based | Works well for structured text | Relies on formatting | HR or legal policies |
| Heading-based | Aligns with policy sections | Requires numbering | SOPs and legal documents |
| Semantic | Groups related sentences | Requires embeddings | Context-sensitive RAG |
| Recursive-char | Handles long text | Not semantic | Large or unstructured documents |

Example usage:
```python
from policy_rag.chunking.alt_chunkers import semantic_chunk
chunks = semantic_chunk(text)
```

---

## LoRA-Enhanced LLM Integration

Implemented under `policy_rag/models/llm_client_lora.py`

This module adds support for Low-Rank Adaptation (LoRA) using the PEFT library, allowing domain-specific fine-tuning of the base LLM.

### Features
- Loads a base model such as `microsoft/phi-1_5` or `tiiuae/falcon-7b`.
- Automatically merges LoRA adapter weights from a local directory.
- Acts as a drop-in replacement for `llm_client.py`.

Example usage:
```python
from policy_rag.models.llm_client_lora import LoRALLMClient

llm = LoRALLMClient(base_model="microsoft/phi-1_5")
answer = llm.generate("What should I do in case of a data breach?")
print(answer)
```

---

## Evaluation with the `evals` Library

Integrated evaluation framework based on OpenAI’s Evals.

**Files:**
```
evals/
 ├── policy_eval.jsonl
 ├── policy_eval.yaml
run_policy_eval.py
```

### Metrics

| Metric | Description |
|---------|-------------|
| Faithfulness | The answer is factually aligned with retrieved context. |
| Grounding | The response cites relevant clauses or evidence. |
| Relevance | The answer addresses the query directly. |
| Completeness | The response covers all aspects of the query. |

Example execution:
```bash
python run_policy_eval.py
```

Example output:
```json
{
  "faithfulness": 0.94,
  "relevance": 0.91,
  "grounding": 0.89,
  "completeness": 0.87,
  "overall_accuracy": 0.90
}
```

Results can optionally be logged to Weights & Biases for experiment tracking.

---

## Unit Test Results

All pipeline components were verified with `pytest`.  
The latest test run achieved a 100% pass rate.

| Test File | Test Name | Status | Description |
|------------|------------|---------|-------------|
| tests/test_chunking.py | test_build_chunks | PASSED | Chunk generation validated |
| tests/test_embedding_and_index.py | test_embeddings_shape | PASSED | Embedding shape verified |
| tests/test_embedding_and_index.py | test_vector_store_add | PASSED | Vector store indexing tested |
| tests/test_eval.py | test_aggregate_eval | PASSED | Evaluation aggregation validated |
| tests/test_generation.py | test_answer_generation | PASSED | LLM response formatting tested |
| tests/test_grounding.py | test_judge_and_enforce | PASSED | Hallucination detection validated |
| tests/test_ingestion.py | test_format_check | PASSED | Format validation tested |
| tests/test_ingestion.py | test_pii_scan | PASSED | PII detection verified |
| tests/test_retrieval.py | test_retrieve | PASSED | Query retrieval tested |

Overall result: 9/9 tests passed successfully.

---

## Summary of System Capabilities

| Capability | Description |
|-------------|-------------|
| Policy ingestion and validation | Extracts and verifies PDF content |
| PII detection | Identifies and classifies sensitive information |
| Configurable chunking | Multiple text segmentation strategies |
| Embedding and retrieval | Context retrieval using vector search |
| LoRA-Enhanced LLM | Domain-specific language adaptation |
| Evals integration | Quantitative RAG evaluation framework |
| Weights & Biases logging | Continuous experiment tracking |
| Full unit test coverage | End-to-end validation of all modules |

---

## License

This project is released under the MIT License.  
You are free to use, modify, and distribute this software with appropriate attribution.
