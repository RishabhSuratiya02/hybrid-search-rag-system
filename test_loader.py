from hybrid_search_rag_system.ingestion.document_loader import load_pdf
from hybrid_search_rag_system.ingestion.text_splitter import split_documents

documents = load_pdf("data/uploads/sample.pdf")

print("total readable pages:", len(documents))

chunks = split_documents(documents)
print("Total chunks:", len(chunks))

for chunk in chunks[:3]:

    print("\nContent:", chunk.page_content[:200])
    print("Metadata:", chunk.metadata)