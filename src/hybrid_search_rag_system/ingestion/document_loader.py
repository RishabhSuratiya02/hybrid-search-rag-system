# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from hybrid_search_rag_system.config import chunk_size, chunk_overlap

# def load_and_split_pdf(pdf_path: str):

#     path = Path(pdf_path).expanduser().resolve()

#     if not path.is_file():
#         raise FileNotFoundError(f"PDF File Not found:{path}")

#     if path.suffix.lower() != ".pdf":
#         raise ValueError("Please Upload a PDF File")

#     try:
#         #load pdf
#         loader = PyPDFLoader(str(path))
#         documents = loader.load()
#     except Exception as exc:
#         raise ValueError(
#             "Could not read the PDF. It may be corrupted or password-protected."
#         ) from exc

#     readable_docs =[]

#     for docs in documents:
#         if not docs.page_content.strip():
#             continue

#         docs.metadata["filename"] = path.name

#         page_index = docs.metadata.get("page")

#         if isinstance(page_index, int):
#             docs.metadata["page_number"] = page_index + 1

#         readable_docs.append(docs)

#     if not readable_docs:
#         raise ValueError(
#              "No readable text was found in this PDF. "
#             "If it is scanned, OCR is required."
#         )
    

#     # split document
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size =chunk_size,
#         chunk_overlap = chunk_overlap
#     )


   
#     chunks = splitter.split_documents(readable_docs)
#     return [
#     chunk for chunk in chunks
#     if chunk.page_content.strip()
# ]

# chunks = load_and_split_pdf("data/uploads/sample.pdf")
# print("Total chunks:", len(chunks))

# for chunk in chunks[:3]:
#     print(chunk.page_content[:200])
#     print(chunk.metadata)




from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):

    path = Path(pdf_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF File Not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please Upload a PDF File")

    try:
        # Load PDF
        loader = PyPDFLoader(str(path))
        documents = loader.load()

    except Exception as exc:
        raise ValueError(
            "Could not read the PDF. "
            "It may be corrupted or password-protected."
        ) from exc

    readable_docs = []

    for doc in documents:

        if not doc.page_content.strip():
            continue

        doc.metadata["filename"] = path.name

        page_index = doc.metadata.get("page")

        if isinstance(page_index, int):
            doc.metadata["page_number"] = page_index + 1

        readable_docs.append(doc)

    if not readable_docs:
        raise ValueError(
            "No readable text was found in this PDF. "
            "If it is scanned, OCR is required."
        )

    return readable_docs
