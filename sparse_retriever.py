
from hybrid_search_rag_system.ingestion.document_loader import load_pdf
from hybrid_search_rag_system.ingestion.text_splitter import split_documents

from hybrid_search_rag_system.retrieval.sparse_retriever import (
    create_bm25_retriever,
    sparse_search
)


# Load PDF
documents = load_pdf("data/uploads/sample.pdf")

# Split documents
chunks = split_documents(documents)

print("Total chunks:", len(chunks))


# Create BM25 index
bm25 = create_bm25_retriever(chunks)


# Search
query = "What are the parts of an essay?"

results = sparse_search(
    query=query,
    chunks=chunks,
    bm25=bm25,
    k=3
)


for index, doc in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("Content:", doc.page_content[:300])
    print("Filename:", doc.metadata.get("filename"))
    print("Page:", doc.metadata.get("page_number"))
    print("Chunk ID:", doc.metadata.get("chunk_id"))
