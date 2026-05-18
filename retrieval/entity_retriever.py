import networkx as nx

def find_relevant_entities(graph, query):
    query_lower = query.lower()
    entities = []

    for node in graph.nodes():
        if node.lower() in query_lower:
            entities.append(node)

    return entities
