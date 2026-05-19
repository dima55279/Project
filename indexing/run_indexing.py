from indexing.loaders import load_documents

from indexing.extraction import extract_graph

from graphdb.ingestion import (
    create_document,
    create_entity,
    create_relationship,
    connect_document
)

from indexing.community_detection import (
    build_communities
)

from indexing.summarization import (
    summarize_community
)

from utils.storage import save_json

from config import (
    DOCS_DIR,
    GRAPH_EXPORT_DIR
)

from tqdm import tqdm

def main():
    # === 1. Загрузка документов ===
    print("📄 Загрузка документов...")
    docs = load_documents(DOCS_DIR)
    print(f"Найдено документов: {len(docs)}")

    # === 2. Обработка документов ===
    print("🔄 Извлечение сущностей и отношений...")
    for doc in tqdm(docs, desc="Обработка документов", unit="doc"):
        # Создание документа
        create_document({
            "name": doc.metadata["source"],
            "content": doc.page_content,
            "filepath": doc.metadata["filepath"]
        })

        extracted = extract_graph(doc.page_content)

        entities = extracted.get("entities", [])
        relations = extracted.get("relationships", [])

        for entity in entities:
            create_entity(entity)
            connect_document(
                entity["id"],
                doc.metadata["source"]
            )

        for rel in relations:
            create_relationship(rel)

    # === 3. Построение сообществ ===
    print("👥 Построение сообществ...")
    communities = build_communities()
    save_json(
        GRAPH_EXPORT_DIR / "communities.json",
        communities
    )

    # === 4. Суммаризация сообществ ===
    print("📝 Суммаризация сообществ...")
    summaries = {}
    for cid, ents in tqdm(communities.items(), desc="Суммаризация сообществ", unit="comm"):
        summaries[cid] = summarize_community(ents)

    save_json(
        GRAPH_EXPORT_DIR / "summaries.json",
        summaries
    )

    print("✅ Индексация завершена!")


if __name__ == "__main__":
    main()
