import networkx as nx

from config import GRAPH_HOPS

def retrieve_subgraph(graph, entities):
    sub_nodes = set()
    
    for entity in entities:
        if entity not in graph:
            continue

        sub_nodes.add(entity)
        current = {entity}

        for _ in range(GRAPH_HOPS):
            next_nodes = set()

            for node in current:
                neighbors = list(graph.neighbors(node))
                next_nodes.update(neighbors)

            sub_nodes.update(next_nodes)
            current = next_nodes

    return graph.subgraph(sub_nodes)
