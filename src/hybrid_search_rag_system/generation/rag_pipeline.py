from hybrid_search_rag_system.retrieval.hybrid_retriever import hybrid_search
from hybrid_search_rag_system.retrieval.reranker import rerank_documents
from hybrid_search_rag_system.generation.llm import get_llm
from hybrid_search_rag_system.generation.prompts import rag_prompt
from hybrid_search_rag_system.utils.citations import (
    validate_citations,
    get_cited_sources
)

def format_context(documents):
    context_parts = []
    sources = []

    for index, doc in enumerate(documents, start=1):
        source_id = f"S{index}"

        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_number", "Unknown")
        chunk_id = doc.metadata.get("chunk_id")

        context_parts.append(
            f"[{source_id}] File: {filename} | Page: {page}\n"
            f"{doc.page_content}"
        )
        sources.append({
            "source_id": source_id,
            "filename": filename,
            "page": page,
            "chunk_id": chunk_id,
            "content": doc.page_content
        })

    context = "\n\n".join(context_parts)

    return context, sources

def run_rag_pipeline(query, chunks, bm25):
    
    if not query.strip():
        raise ValueError("Question cannot be empty.")

    # Hybrid Retrieval
    hybrid_results = hybrid_search(
        query=query,
        chunks=chunks,
        bm25=bm25,
        dense_k=10,
        sparse_k=10,
        top_k=10
    )

     # Cross-Encoder Reranking
    reranked_results = rerank_documents(
        query=query,
        documents=hybrid_results,
        top_k=4
    )

    if not reranked_results:
        return {
            "answer": "I could not find this information in the uploaded document.",
            "sources": []
        }

    context, sources = format_context(reranked_results)

    llm = get_llm()

     # Create prompt
    messages = rag_prompt.format_messages(
        context=context,
        question=query
    )

    # Generate answer
    
    response = llm.invoke(messages)

    answer = response.content

    # Validate citations
    citation_validation = validate_citations(
        answer=answer,
        sources=sources
    )

    cited_sources = get_cited_sources(
        answer=answer,
        sources=sources
    )

    return {
        "answer": answer,
        "sources": sources,
        "cited_sources": cited_sources,
        "citation_validation": citation_validation
    }
