
import streamlit as st
import requests


# -------------------------------
# Configuration
# -------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Hybrid Search RAG",
    page_icon="📚",
    layout="wide"
)


# -------------------------------
# Session State
# -------------------------------

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.title("📚 Hybrid Search RAG")

    st.caption(
        "Dense Search + BM25 + RRF + Cross-Encoder"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "Process PDF",
            use_container_width=True
        ):

            with st.spinner("Uploading and indexing PDF..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=300
                    )

                    response.raise_for_status()

                    data = response.json()

                    st.session_state.document_id = data["document_id"]
                    st.session_state.filename = data["filename"]
                    st.session_state.messages = []

                    st.success("PDF processed successfully!")

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"PDF upload failed: {exc}"
                    )

    st.divider()

    if st.session_state.document_id:

        st.success("Document ready")

        st.write(
            f"**File:** {st.session_state.filename}"
        )

        if st.button(
            "Clear Chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("Upload and process a PDF to start.")


# -------------------------------
# Main Interface
# -------------------------------

st.title("Hybrid Search RAG Assistant")

st.caption(
    "Ask questions from your PDF using "
    "Dense Retrieval, BM25 and Cross-Encoder Reranking."
)

st.divider()


# -------------------------------
# Display Chat History
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander(
                    f"View Sources ({len(sources)})"
                ):

                    for source in sources:

                        st.markdown(
                            f"**[{source['source_id']}] "
                            f"{source['filename']} "
                            f"— Page {source['page']}**"
                        )

                        st.caption(
                            source.get("content", "")
                        )

            validation = message.get(
                "citation_validation", {}
            )

            if validation and not validation.get(
                "is_valid", False
            ):

                st.warning(
                    "Some citations may be missing or invalid. "
                    "Verify the answer against the retrieved passages."
                )


# -------------------------------
# Chat Input
# -------------------------------

question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=not bool(st.session_state.document_id)
)


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching and generating answer..."):

            try:

                payload = {
                    "document_id": st.session_state.document_id,
                    "question": question
                }

                response = requests.post(
                    f"{API_URL}/ask",
                    json=payload,
                    timeout=180
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer generated."
                )

                sources = data.get("sources", [])

                validation = data.get(
                    "citation_validation", {}
                )

                st.markdown(answer)

                if sources:

                    with st.expander(
                        f"View Sources ({len(sources)})"
                    ):

                        for source in sources:

                            st.markdown(
                                f"**[{source['source_id']}] "
                                f"{source['filename']} "
                                f"— Page {source['page']}**"
                            )

                            st.caption(
                                source.get("content", "")
                            )

                if validation and not validation.get(
                    "is_valid", False
                ):

                    st.warning(
                        "Some citations may be missing or invalid. "
                        "Verify the answer against the retrieved passages."
                    )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "citation_validation": validation
                })

            
            except requests.exceptions.HTTPError as exc:

                try:
                    error_detail = response.json().get(
                        "detail",
                        str(exc)
                    )
                except ValueError:
                    error_detail = response.text or str(exc)

                st.error(f"Backend Error: {error_detail}")

            except requests.exceptions.RequestException as exc:

                st.error(f"Connection Error: {exc}")

