import os 
from dotenv import load_dotenv
from pathlib import Path

# project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

#load api key
load_dotenv(BASE_DIR/ ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# llm model integrate

llm_model = "openai/gpt-oss-20b"
temperature  = 0 
max_output_tokens = 800

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1000
chunk_overlap = 200

DENSE_RETRIEVAL_K = 10
CHROMA_COLLECTION_NAME = "hybrid_rag_documents"


SPARSE_RETRIEVAL_K = 10

RRF_K = 60


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_TOP_K = 4

# Storage paths
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
BM25_DIR = BASE_DIR / "data" / "bm25"



def validate_config() -> None:
    """Validate required configuration values."""

    if not GROQ_API_KEY or not GROQ_API_KEY.strip():
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be positive.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be between 0 and CHUNK_SIZE."
        )

    if min(
        DENSE_RETRIEVAL_K,
        SPARSE_RETRIEVAL_K,
        RERANK_TOP_K,
        RRF_K
    ) <= 0:
        raise ValueError(
            "Retrieval settings must be positive."
        )


def create_directories() -> None:
    """Create required storage directories."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    BM25_DIR.mkdir(parents=True, exist_ok=True)


