import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_search_rag_system.config import chunk_size, chunk_overlap

def split_documents(documents: list[Document])-> list[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    processed_chunks = []

    for index, chunk in enumerate(chunks):
        if not chunk.page_content.strip():
            continue

        # generate unique chunk id
        content = (
            f"{chunk.metadata.get('source','')}|"
            f"{chunk.metadata.get('page','')}|"
            f"{index}|"
            f"{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        # add metadata
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = index

        processed_chunks.append(chunk)

    return processed_chunks

