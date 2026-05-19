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



def main():

    docs = load_documents(DOCS_DIR)

    for doc in docs:

        create_document({
            "name": doc.metadata["source"],
            "content": doc.page_content,
            "filepath": doc.metadata["filepath"]
        })

        extracted = extract_graph(
            doc.page_content
        )

        entities = extracted.get(
            "entities",
            []
        )

        relations = extracted.get(
            "relationships",
            []
        )

        for entity in entities:

            create_entity(entity)

            connect_document(
                entity["id"],
                doc.metadata["source"]
            )

        for rel in relations:

            create_relationship(rel)

    communities = build_communities()

    save_json(
        GRAPH_EXPORT_DIR / "communities.json",
        communities
    )

    summaries = {}

    for cid, ents in communities.items():

        summaries[cid] = summarize_community(
            ents
        )

    save_json(
        GRAPH_EXPORT_DIR / "summaries.json",
        summaries
    )


if __name__ == "__main__":
    main()
