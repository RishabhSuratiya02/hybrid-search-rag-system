
from pathlib import Path
from uuid import uuid4
import shutil
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from hybrid_search_rag_system.config import UPLOAD_DIR

from hybrid_search_rag_system.ingestion.document_loader import load_pdf
from hybrid_search_rag_system.ingestion.text_splitter import split_documents

from hybrid_search_rag_system.retrieval.dense_retriever import (
    add_documents
)

from hybrid_search_rag_system.retrieval.sparse_retriever import (
    create_bm25_retriever
)

from hybrid_search_rag_system.generation.rag_pipeline import (
    run_rag_pipeline
)


# Initialize FastAPI
app = FastAPI(
    title="Hybrid Search RAG API",
    description="PDF Question Answering using Dense Search, BM25, RRF and Cross-Encoder Reranking",
    version="1.0.0"
)

# Create upload directory
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory document sessions
document_sessions = {}


# Request schema
class QuestionRequest(BaseModel):
    document_id: str
    question: str


# Health endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "Hybrid Search RAG API is running"
    }


# PDF upload endpoint
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    document_id = str(uuid4())

    # Unique filename prevents overwriting uploads
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

    try:
        # Save uploaded PDF
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = load_pdf(str(pdf_path))

        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("No readable chunks found.")

        # Store original filename for citations
        for chunk in chunks:
            chunk.metadata["filename"] = filename
            chunk.metadata["document_id"] = document_id

        add_documents(chunks)

        bm25 = create_bm25_retriever(chunks)

        document_sessions[document_id] = {
            "filename": filename,
            "chunks": chunks,
            "bm25": bm25
        }

        return {
            "message": "PDF uploaded and indexed successfully.",
            "document_id": document_id,
            "filename": filename,
            "total_pages": len(documents),
            "total_chunks": len(chunks)
        }

    except Exception as exc:

        # Remove incomplete uploaded file
        if pdf_path.exists():
            pdf_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(exc)}"
        )

    finally:
        file.file.close()


# Question-answering endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):

    session = document_sessions.get(request.document_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Document session not found. Upload the PDF again."
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = run_rag_pipeline(
            query=request.question,
            chunks=session["chunks"],
            bm25=session["bm25"],
            document_id=request.document_id
        )

        return {
            "document_id": request.document_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "cited_sources": result.get("cited_sources", []),
            "citation_validation": result.get(
                "citation_validation",
                {}
            )
        }

    
    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(exc)}"
        )

