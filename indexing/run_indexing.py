from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rich.console import Console

from indexing.loaders import load_documents
from indexing.extraction import extract_graph_batch
from graphdb.ingestion import (
    create_documents_batch, create_entities_batch,
    create_relationships_batch, create_facts_batch, connect_documents_batch
)
from indexing.community_detection import build_communities
from indexing.summarization import summarize_community
from utils.storage import save_json
from config import DOCS_DIR, GRAPH_EXPORT_DIR, BATCH_SIZE, NUM_WORKERS

console = Console()


def process_batch(batch_docs):
    texts = [doc.page_content for doc in batch_docs]
    batch_meta = [{
        "name": doc.metadata["source"],
        "content": doc.page_content,
        "filepath": doc.metadata["filepath"]
    } for doc in batch_docs]

    extracted = extract_graph_batch(texts, batch_names=[m["name"] for m in batch_meta])
    
    entities = extracted.get("entities", [])
    relations = extracted.get("relationships", [])
    facts = extracted.get("facts", [])

    connections = [
        {"entity": e["id"], "document": meta["name"]}
        for e in entities for meta in batch_meta
    ]

    return {
        "documents": batch_meta,
        "entities": entities,
        "relations": relations,
        "facts": facts,
        "connections": connections
    }


def main():
    print("🚀 Запуск Legal GraphRAG Indexing...")

    docs = load_documents(DOCS_DIR)
    if not docs:
        print("❌ Документы не найдены!")
        return

    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    
    all_entities = []
    all_relations = []
    all_facts = []
    all_documents = []
    all_connections = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Извлечение"):
            result = future.result()
            all_documents.extend(result["documents"])
            all_entities.extend(result["entities"])
            all_relations.extend(result["relations"])
            all_facts.extend(result["facts"])
            all_connections.extend(result["connections"])

    print("💾 Сохранение в Neo4j...")
    create_documents_batch(all_documents)
    create_entities_batch(all_entities)
    create_relationships_batch(all_relations)
    create_facts_batch(all_facts)
    connect_documents_batch(all_connections)

    print("👥 Построение сообществ...")
    communities = build_communities()
    save_json(GRAPH_EXPORT_DIR / "communities.json", communities)

    print("📝 Суммаризация...")
    summaries = {}
    for cid, ents in tqdm(communities.items(), desc="Суммаризация"):
        summaries[cid] = summarize_community(ents)
    
    save_json(GRAPH_EXPORT_DIR / "summaries.json", summaries)

    print(f"🎉 Индексация завершена! Сущностей: {len(all_entities)} | Фактов: {len(all_facts)}")


if __name__ == "__main__":
    main()