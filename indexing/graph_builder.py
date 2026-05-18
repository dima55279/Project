import networkx as nx

def build_graph(extractions):
    graph = nx.MultiDiGraph()
    
    for item in extractions:
        chunk_id = item["chunk_id"]
        entities = item["entities"]
        relationships = item["relationships"]

        for entity in entities:
            graph.add_node(
                entity["id"],
                entity_type=entity.get("type"),
                description=entity.get("description"),
                chunk_id=chunk_id
            )

        for rel in relationships:
            graph.add_edge(
                rel["source"],
                rel["target"],
                relation=rel.get("relation"),
                description=rel.get("description"),
                chunk_id=chunk_id
            )

    return graph
