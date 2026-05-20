def build_citations(evidence):

    documents = []

    fragments = []

    for e in evidence:

        doc = e.get("document")

        content = e.get("content")

        if doc:

            documents.append(doc)

        if content:

            fragments.append(
                content[:1000]
            )

    return {
        "documents": list(set(documents)),
        "evidence": fragments
    }
