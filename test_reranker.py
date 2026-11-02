
from hybrid_search_rag_system.ingestion.document_loader import load_pdf
from hybrid_search_rag_system.ingestion.text_splitter import split_documents

from hybrid_search_rag_system.retrieval.dense_retriever import add_documents

from hybrid_search_rag_system.retrieval.sparse_retriever import (
    create_bm25_retriever
)

from hybrid_search_rag_system.retrieval.hybrid_retriever import hybrid_search

from hybrid_search_rag_system.retrieval.reranker import rerank_documents


# Step 1: Load PDF
documents = load_pdf("data/uploads/sample.pdf")

# Step 2: Split into chunks
chunks = split_documents(documents)

# Step 3: Store chunks in ChromaDB
add_documents(chunks)

# Step 4: Create BM25 index
bm25 = create_bm25_retriever(chunks)

# Step 5: Hybrid Search
query = "What is a thesis statement?"

hybrid_results = hybrid_search(
    query=query,
    chunks=chunks,
    bm25=bm25,
    dense_k=10,
    sparse_k=10,
    top_k=10
)

print("Hybrid Results:", len(hybrid_results))

# Step 6: Rerank
reranked_results = rerank_documents(
    query=query,
    documents=hybrid_results,
    top_k=4
)

print("Reranked Results:", len(reranked_results))

# Step 7: Display final results
for index, doc in enumerate(reranked_results, start=1):

    print(f"\nResult {index}")
    print("Content:", doc.page_content[:300])
    print("Filename:", doc.metadata.get("filename"))
    print("Page:", doc.metadata.get("page_number"))
    print("Chunk ID:", doc.metadata.get("chunk_id"))
