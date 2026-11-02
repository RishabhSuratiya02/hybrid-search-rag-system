
import re


# Step 1: Extract citation labels from LLM answer
def extract_citations(answer):

    citations = re.findall(
        r"\[S\d+\]",
        answer
    )

    # Remove duplicate labels
    return list(dict.fromkeys(citations))


# Step 2: Validate citation labels
def validate_citations(answer, sources):

    # Extract citations from answer
    used_citations = extract_citations(answer)

    # Get available source labels
    available_citations = [
        f"[{source['source_id']}]"
        for source in sources
    ]

    valid_citations = []
    invalid_citations = []

    for citation in used_citations:

        if citation in available_citations:
            valid_citations.append(citation)
        else:
            invalid_citations.append(citation)

    return {
        "used_citations": used_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "has_citations": len(used_citations) > 0,
        "is_valid": len(invalid_citations) == 0 and len(valid_citations) > 0
    }


# Step 3: Get cited source details
def get_cited_sources(answer, sources):

    used_citations = extract_citations(answer)

    cited_sources = []

    for source in sources:

        label = f"[{source['source_id']}]"

        if label in used_citations:
            cited_sources.append(source)

    return cited_sources
