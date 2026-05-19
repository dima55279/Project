from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import math

from indexing.loaders import load_documents
from indexing.extraction import extract_graph_batch
from graphdb.ingestion import (
    create_documents_batch,
    create_entities_batch,
    create_relationships_batch,
    connect_documents_batch
)
from indexing.community_detection import build_communities
from indexing.summarization import summarize_community
from utils.storage import save_json
from config import DOCS_DIR, GRAPH_EXPORT_DIR, BATCH_SIZE, NUM_WORKERS


def process_batch(batch_docs):
    """Обрабатывает батч документов"""
    texts = [doc.page_content for doc in batch_docs]
    batch_meta = [{
        "name": doc.metadata["source"],
        "content": doc.page_content,
        "filepath": doc.metadata["filepath"]
    } for doc in batch_docs]

    # Извлечение
    extracted = extract_graph_batch(texts)
    
    entities = extracted.get("entities", [])
    relations = extracted.get("relationships", [])

    return {
        "documents": batch_meta,
        "entities": entities,
        "relations": relations,
        "connections": [
            {"entity": e["id"], "document": meta["name"]}
            for e in entities for meta in batch_meta
        ]
    }


def main():
    print("🚀 Запуск оптимизированной индексации...")

    # 1. Загрузка
    docs = load_documents(DOCS_DIR)
    print(f"📄 Найдено документов: {len(docs)}")

    # 2. Batch processing
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    
    all_entities = []
    all_relations = []
    all_connections = []
    all_documents = []

    print(f"🔄 Обработка {len(batches)} батчей по {BATCH_SIZE} документов...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Извлечение графов"):
            result = future.result()
            all_documents.extend(result["documents"])
            all_entities.extend(result["entities"])
            all_relations.extend(result["relations"])
            all_connections.extend(result["connections"])

    # 3. Массовое сохранение в Neo4j
    print("💾 Сохранение в Neo4j...")
    create_documents_batch(all_documents)
    create_entities_batch(all_entities)
    create_relationships_batch(all_relations)
    connect_documents_batch(all_connections)

    # 4. Сообщества и суммаризация
    print("👥 Построение сообществ...")
    communities = build_communities()
    save_json(GRAPH_EXPORT_DIR / "communities.json", communities)

    print("📝 Суммаризация сообществ...")
    summaries = {}
    for cid, ents in tqdm(communities.items(), desc="Суммаризация"):
        summaries[cid] = summarize_community(ents)
    
    save_json(GRAPH_EXPORT_DIR / "summaries.json", summaries)

    print("🎉 Индексация успешно завершена!")
    print(f"   Сущностей: {len(all_entities)} | Отношений: {len(all_relations)}")


if __name__ == "__main__":
    main()