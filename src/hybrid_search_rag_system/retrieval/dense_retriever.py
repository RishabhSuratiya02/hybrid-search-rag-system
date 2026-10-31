from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from hybrid_search_rag_system.config import(
    embedding_model,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    DENSE_RETRIEVAL_K
)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name = embedding_model
)

# Initialize ChromaDB
vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function= embeddings,
    persist_directory=str(CHROMA_DIR)
) 

# Store chunks in ChromaDB

def add_documents(chunks):

    if not chunks:
        return []

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunks ID found.")

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )
    return ids


def dense_search(query, k = DENSE_RETRIEVAL_K):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    results = vectorstore.similarity_search(
        query=query,
        k = k
    )

    return results