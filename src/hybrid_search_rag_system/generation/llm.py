from langchain_groq import ChatGroq

from hybrid_search_rag_system.config import (
    GROQ_API_KEY,
    llm_model,
    temperature,
    max_output_tokens
)

def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    llm = ChatGroq(
        model= llm_model,
        temperature=temperature,
        max_output_tokens = max_output_tokens,
        api_key=GROQ_API_KEY
    )

    return llm 