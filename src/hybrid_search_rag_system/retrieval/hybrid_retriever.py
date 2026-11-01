from hybrid_search_rag_system.retrieval.dense_retriever import dense_search

from hybrid_search_rag_system.retrieval.sparse_retriever import (
    create_bm25_retriever,
    sparse_search
)

#RRF function

def reciprocal_rank_fusion(dense_results, sparse_results, rrf_k =60):

    scores ={}
    documents = {}

    for results in [dense_results, sparse_results]:
        for rank, doc in enumerate(results, start=1):
            chunk_id = doc.metadata["chunk_id"]

            if chunk_id not in scores:
                scores[chunk_id]=0
                documents[chunk_id] = doc

            scores[chunk_id] +=1/(rrf_k + rank)


    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True
    )

    return [documents[chunk_id] for chunk_id in ranked_ids]


def hybrid_search(query, chunks, bm25, dense_k=10, sparse_k=10, top_k=5):
    
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("Query cannot be empty.")

    dense_results = dense_search(
        query=query,
        k = dense_k
    )

    sparse_results =  sparse_search(
        query= query,
        chunks= chunks,
        bm25=bm25,
        k= sparse_k
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )

    return fused_results