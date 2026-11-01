from rank_bm25 import BM25Okapi
from hybrid_search_rag_system.config import SPARSE_RETRIEVAL_K

# create bm25retriever

def create_bm25_retriever(chunks):
    if not chunks:
        raise ValueError("No chunks Provided.")

    tokenized_corpus =[
        chunk.page_content.lower().split()
        for chunk in chunks
    ] 

    # create bm25 index 
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

def sparse_search(query,chunks,bm25, k=SPARSE_RETRIEVAL_K):

    if not query.strip():
        raise ValueError("query cannot be empty")

    if k <=0:
        raise ValueError("K must be positive")

    tokenized_query= query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key = lambda i: scores[i],
        reverse= True
    )

    results = [
        chunks[i]
        for i in ranked_indices
        if scores[i]>0
    ]
    return results
