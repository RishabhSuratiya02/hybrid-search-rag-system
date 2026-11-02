from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(query, documents, top_k =4):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not documents:
        return []

    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = reranker_model.predict(pairs)

    scored_documents = list(zip(documents, scores))
    scored_documents.sort(
        key=lambda item: float(item[1]),
        reverse=True
    )

    reranked_documents = [
    doc
    for doc, score in scored_documents[:top_k]
    ]

    return reranked_documents