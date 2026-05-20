# agents/citation_agent.py

def build_citations(evidence):
    documents = []
    snippets = []

    for item in evidence:
        doc_name = item.get("document")
        if doc_name:
            documents.append(doc_name)
        if item.get("content"):
            snippets.append(item["content"][:700])

    # Убираем дубликаты, сохраняя порядок
    documents = list(dict.fromkeys(documents))

    return {
        "documents": documents,
        "evidence": snippets
    }