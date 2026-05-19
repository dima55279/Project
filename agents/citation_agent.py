def build_citations(evidence):

    documents = []

    snippets = []

    for item in evidence:

        documents.append(
            item["document"]
        )

        snippets.append(
            item["content"][:500]
        )

    return {
        "documents": list(set(documents)),
        "evidence": snippets
    }
