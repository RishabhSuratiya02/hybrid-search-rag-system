# Hybrid Search RAG System

A PDF question-answering application built with **Python, FastAPI, and Streamlit**. It combines semantic search with keyword-based retrieval, merges results using Reciprocal Rank Fusion (RRF), and reranks the retrieved passages before generating an answer with a Groq-hosted LLM.

The goal of this project is to explore how different retrieval methods can work together in a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- Upload a PDF and extract readable text.
- Split documents into overlapping chunks with page metadata.
- Store embeddings in **ChromaDB** for dense (semantic) retrieval.
- Use **BM25** for keyword-based sparse retrieval.
- Combine dense and sparse rankings with **Reciprocal Rank Fusion (RRF)**.
- Rerank candidates using a **Cross-Encoder** model.
- Generate document-grounded answers using **Groq** and a configurable LLM.
- Show retrieved source passages, PDF page numbers, and citation labels such as `[S1]`.
- Check whether generated citation labels correspond to retrieved sources.
- Expose PDF upload and question-answering endpoints through **FastAPI**.
- Provide a chat interface through **Streamlit**.

> **Citation limitation:** Citation validation checks whether a label exists in the retrieved source list. It does not independently verify that every claim is supported by the cited passage.

## How It Works

```text
PDF upload
    |
    v
PDF loading and text extraction
    |
    v
Text chunking + metadata
    |
    +-----------------------------+
    |                             |
    v                             v
Dense index (ChromaDB)        BM25 index
    |                             |
    v                             v
Semantic search              Keyword search
    |                             |
    +-------------+---------------+
                  |
                  v
        Reciprocal Rank Fusion
                  |
                  v
          Cross-Encoder reranking
                  |
                  v
       Top passages + source labels
                  |
                  v
           Groq-hosted LLM
                  |
                  v
       Answer + citations + sources
```

### Retrieval Pipeline

1. **Dense retrieval:** Hugging Face sentence embeddings are stored in ChromaDB and used to find semantically similar chunks.
2. **Sparse retrieval:** BM25 ranks chunks by keyword relevance.
3. **Rank fusion:** RRF combines both ranked lists without requiring their scores to share the same scale.
4. **Reranking:** A Cross-Encoder evaluates query–passage pairs and reorders the candidate chunks.
5. **Generation:** The highest-ranked passages are passed to the LLM with instructions to answer from the provided context and cite source labels.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Language | Python |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| RAG framework | LangChain |
| Embeddings | Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB |
| Sparse retrieval | `rank-bm25` |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| LLM provider | Groq API |
| PDF processing | PyPDFLoader / pypdf |

## Project Structure

```text
hybrid-search-rag-system/
├── app.py                         # Streamlit frontend
├── main.py                        # FastAPI backend
├── requirements.txt
├── .env                            # Local secrets (do not commit)
├── .gitignore
├── data/
│   ├── uploads/                   # Uploaded PDFs
│   ├── chroma/                    # Persistent vector store
│   └── bm25/                      # Reserved for BM25 storage
├── src/
│   └── hybrid_search_rag_system/
│       ├── __init__.py
│       ├── config.py
│       ├── ingestion/
│       │   ├── document_loader.py
│       │   └── text_splitter.py
│       ├── retrieval/
│       │   ├── dense_retriever.py
│       │   ├── sparse_retriever.py
│       │   ├── hybrid_retriever.py
│       │   └── reranker.py
│       ├── generation/
│       │   ├── llm.py
│       │   ├── prompts.py
│       │   └── rag_pipeline.py
│       └── utils/
│           └── citations.py
├── evaluation/                    # Evaluation work (planned)
└── tests/                         # Tests (as added)
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the placeholders with your actual repository URL and folder name.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For a `src/` layout, ensure the package is importable. If the project is not installed as a package, set `PYTHONPATH` to `src` before running it:

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "src"
```

**macOS / Linux:**

```bash
export PYTHONPATH=src
```

### 4. Configure the API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit `.env` or your real API key. The configured Groq model must be available to your account.

### 5. Start the FastAPI backend

In the first terminal:

```bash
python -m uvicorn main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

Health endpoint: http://127.0.0.1:8000/health

### 6. Start the Streamlit frontend

In a **second terminal** (with the same environment and `PYTHONPATH` setup):

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a PDF, click **Process PDF**, and ask a question about the document.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check API status |
| `POST` | `/upload` | Upload and index a PDF; returns a `document_id` |
| `POST` | `/ask` | Ask a question about a previously uploaded PDF |

**Example `/ask` request:**

```json
{
  "document_id": "<document-id-from-upload>",
  "question": "What is a thesis statement?"
}
```

The response includes an answer, source passages, and citation-validation metadata.

## Example Use Case

1. Upload an academic PDF.
2. Ask: **“What is a thesis statement?”**
3. The system retrieves relevant passages through dense and BM25 search.
4. RRF combines the rankings, and the Cross-Encoder selects the strongest candidates.
5. The LLM produces an answer grounded in the retrieved text and includes source labels when possible.

The exact answer depends on the uploaded PDF and model output.

## Current Limitations

- **BM25 and document sessions are held in memory.** Restarting the backend loses the active document sessions, even though ChromaDB may retain vectors.
- **Citation labels are checked, not fact-verified.** A valid label does not guarantee the attached claim is supported.
- **PDF text extraction is not OCR.** Image-only/scanned PDFs may not work without an OCR step.
- **Retrieval quality has not yet been benchmarked.** No Recall@K or MRR@K improvements are claimed.
- **Local development setup:** Authentication, upload limits, persistent session management, and production-grade error handling still need work.

## Planned Improvements

- Evaluate dense-only, BM25-only, hybrid, and hybrid + reranking using a labeled question set.
- Report retrieval metrics such as Recall@K and MRR@K.
- Add stronger citation support checks and no-answer evaluation.
- Persist document/index metadata across backend restarts.
- Add file validation, upload limits, and deployment configuration.

## What I Learned

This project helped me implement and connect several retrieval approaches in one RAG application. It also highlighted practical challenges such as document isolation, ranking quality, source attribution, API integration, and handling unsupported questions.

## License

No license is specified yet. Add a `LICENSE` file if you want to define how others may use or modify the project.
